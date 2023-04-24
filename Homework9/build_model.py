import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import nltk
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
import json
import pickle

# Settings
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
model_save_dir = "data/model"
corpus_name = "southpark-corpus"

# ------------------------------------------------------------------------------------------------------------

#
# Preprocess data
#

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

# Write tokens to file
tokensf = open("data/testing/tokens.txt", "w", encoding="UTF-8") 

# Create vocabulary and vectorize
class Vocab:
  def __init__(self):
    self.trimmed = False
    self.word_to_index = {}
    self.word_to_count = {}
    self.index_to_word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
    self.num_words = 3  # Count SOS, EOS, PAD

  def loadData (self):
    print('Opening text to build vocab...')
    with open('data/southpark-corpus.txt', 'r', encoding='UTF-8') as f:
      text = f.readlines()
      print('Building vocabulary...')
      for line in text:
        line = line.replace("\n","")
        tokens = line.split(" ")
        tokensf.write(str(tokens))
        for token in tokens:
          if token not in self.word_to_index:
            self.word_to_index[token] = self.num_words
            self.word_to_count[token] = 1
            self.index_to_word[self.num_words] = token
            self.num_words += 1
          else:
            self.word_to_count[token] += 1

# Make vocab and load data
vocab = Vocab()
vocab.loadData()

# Write word_to_index to file
with open("data/testing/word_to_index.txt", "w", encoding="UTF-8") as f:
  f.write(str(vocab.word_to_index))
  f.close()
tokensf.close()

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
  )

# Get Pairs
MAX_PAIR_LENGTH = 15
with open('data/conversations.txt', 'r', encoding='UTF-8') as f:
  lines = f.readlines()
  pairs = [[re.sub(r"[ ]{2,}", " ", re.sub(r"[\n]", "", s)) for s in l.split('\t')] for l in lines]
  pairs = [[unicodeToAscii(s.lower().strip()) for s in pair] for pair in pairs]
  # Filter pairs to only those that have a token length less than MAX_PAIR_LENGTH
  pairs = [pair for pair in pairs if len(pair[0].split(' ')) < MAX_PAIR_LENGTH and len(pair[1].split(' ')) < MAX_PAIR_LENGTH]

print( pairs[:10] )

# ------------------------------------------------------------------------------------------------------------


#
# Prepare data for model
#

def indexesFromSentence(vocab, sentence):
  return [vocab.word_to_index[word] for word in sentence.split(" ")] + [EOS_token]

def zeroPadding(l, fillvalue=PAD_token):
  return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(line, value=PAD_token):
  m = []
  for i, seq in enumerate(line):
    m.append([])
    for token in seq:
      if token == PAD_token:
        m[i].append(0)
      else:
        m[i].append(1)
  return m

# Returns padded input sequence tensor and lengths
def getInputVar(l, vocab):
  indexes_batch = [indexesFromSentence(vocab, sentence) for sentence in l]
  lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
  padded_list = zeroPadding(indexes_batch)
  padded_var = torch.LongTensor(padded_list)
  return padded_var, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def getOutputVar(l, vocab):
  indexes_batch = [indexesFromSentence(vocab, sentence) for sentence in l]
  max_target_len = max([len(indexes) for indexes in indexes_batch])
  padded_list = zeroPadding(indexes_batch)
  mask = binaryMatrix(padded_list)
  mask = torch.BoolTensor(mask)
  padded_var = torch.LongTensor(padded_list)
  return padded_var, mask, max_target_len

# Returns all items for a given batch of pairs
def batchToTrainData(vocab, pair_batch):
  pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
  input_batch, output_batch = [], []
  for pair in pair_batch:
    input_batch.append(pair[0])
    output_batch.append(pair[1])
  inp, lengths = getInputVar(input_batch, vocab)
  output, mask, max_target_len = getOutputVar(output_batch, vocab)
  return inp, lengths, output, mask, max_target_len


# Example for validation
small_batch_size = 5
batches = batchToTrainData(vocab, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)

# ------------------------------------------------------------------------------------------------------------

#
# Define model
#

# Encoder
# encode variable length input sequence to a fixed-length context vector

class EncoderRNN(nn.Module):
  def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
    super(EncoderRNN, self).__init__()
    self.n_layers = n_layers
    self.hidden_size = hidden_size
    self.embedding = embedding

    # Initialize GRU; the input_size and hidden_size parameters are both set to 'hidden_size'
    #   because our input size is a word embedding with number of features == hidden_size
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

  def forward(self, input_seq, input_lengths, hidden=None):
    # Convert word indexes to embeddings
    embedded = self.embedding(input_seq)
    # Pack padded batch of sequences for RNN module
    packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
    # Forward pass through GRU
    outputs, hidden = self.gru(packed, hidden)
    # Unpack padding
    outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
    # Sum bidirectional GRU outputs
    outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
    # Return output and final hidden state
    return outputs, hidden

# Decoder
# take input word and context vector, and return a guess for the next word in the sequence and a hidden state for the next iteration

# Luong attention layer
class Attn(nn.Module):
  def __init__(self, method, hidden_size):
    super(Attn, self).__init__()
    self.method = method
    if self.method not in ['dot', 'general', 'concat']:
      raise ValueError(self.method, "is not an appropriate attention method.")
    self.hidden_size = hidden_size
    if self.method == 'general':
      self.attn = nn.Linear(self.hidden_size, hidden_size)
    elif self.method == 'concat':
      self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
      self.v = nn.Parameter(torch.FloatTensor(hidden_size))

  def dot_score(self, hidden, encoder_output):
    return torch.sum(hidden * encoder_output, dim=2)

  def general_score(self, hidden, encoder_output):
    energy = self.attn(encoder_output)
    return torch.sum(hidden * energy, dim=2)

  def concat_score(self, hidden, encoder_output):
    energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
    return torch.sum(self.v * energy, dim=2)

  def forward(self, hidden, encoder_outputs):
    # Calculate the attention weights (energies) based on the given method
    if self.method == 'general':
      attn_energies = self.general_score(hidden, encoder_outputs)
    elif self.method == 'concat':
      attn_energies = self.concat_score(hidden, encoder_outputs)
    elif self.method == 'dot':
      attn_energies = self.dot_score(hidden, encoder_outputs)

    # Transpose max_length and batch_size dimensions
    attn_energies = attn_energies.t()

    # Return the softmax normalized probability scores (with added dimension)
    return F.softmax(attn_energies, dim=1).unsqueeze(1)
  
class LuongAttnDecoderRNN(nn.Module):
  def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
    super(LuongAttnDecoderRNN, self).__init__()

    # Keep for reference
    self.attn_model = attn_model
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.n_layers = n_layers
    self.dropout = dropout

    # Define layers
    self.embedding = embedding
    self.embedding_dropout = nn.Dropout(dropout)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
    self.concat = nn.Linear(hidden_size * 2, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

    self.attn = Attn(attn_model, hidden_size)

  def forward(self, input_step, last_hidden, encoder_outputs):
    # Note: we run this one step (word) at a time
    # Get embedding of current input word
    embedded = self.embedding(input_step)
    embedded = self.embedding_dropout(embedded)
    # Forward through unidirectional GRU
    rnn_output, hidden = self.gru(embedded, last_hidden)
    # Calculate attention weights from the current GRU output
    attn_weights = self.attn(rnn_output, encoder_outputs)
    # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
    context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
    # Concatenate weighted context vector and GRU output using Luong eq. 5
    rnn_output = rnn_output.squeeze(0)
    context = context.squeeze(1)
    concat_input = torch.cat((rnn_output, context), 1)
    concat_output = torch.tanh(self.concat(concat_input))
    # Predict next word using Luong eq. 6
    output = self.out(concat_output)
    output = F.softmax(output, dim=1)
    # Return output and final hidden state
    return output, hidden


# ------------------------------------------------------------------------------------------------------------

#
# Define training procedure
#

# Must define a custom loss function to calculate loss based on the decoder's output tensor, and binary mask tensor describing the padding of such
# Loss is calculated as the average negative log likelihood of the elements that are 1 in the mask tensor
def maskNLLLoss(inp, target, mask):
  nTotal = mask.sum()
  crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
  loss = crossEntropy.masked_select(mask).mean()
  loss = loss.to(device)
  return loss, nTotal.item()

# Function for a single training iteration - a singular batch
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_PAIR_LENGTH):
  # Zero gradients
  encoder_optimizer.zero_grad()
  decoder_optimizer.zero_grad()

  # Set device options
  input_variable = input_variable.to(device)
  target_variable = target_variable.to(device)
  mask = mask.to(device)
  # Lengths for RNN packing should always be on the CPU
  lengths = lengths.to("cpu")

  # Initialize variables
  loss = 0
  print_losses = []
  n_totals = 0

  # Forward pass through encoder
  encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

  # Create initial decoder input (start with SOS tokens for each sentence)
  decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
  decoder_input = decoder_input.to(device)

  # Set initial decoder hidden state to the encoder's final hidden state
  decoder_hidden = encoder_hidden[:decoder.n_layers]

  # Determine if we are using teacher forcing this iteration
  use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

  # Forward batch of sequences through decoder one time step at a time
  if use_teacher_forcing:
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden, encoder_outputs )
      # Teacher forcing: next input is current target
      decoder_input = target_variable[t].view(1, -1)
      # Calculate and accumulate loss
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
      loss += mask_loss
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal
  else:
    for t in range(max_target_len):
      decoder_output, decoder_hidden = decoder( decoder_input, decoder_hidden, encoder_outputs )
      # No teacher forcing: next input is decoder's own current output
      _, topi = decoder_output.topk(1)
      decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
      decoder_input = decoder_input.to(device)
      # Calculate and accumulate loss
      mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
      loss += mask_loss
      print_losses.append(mask_loss.item() * nTotal)
      n_totals += nTotal

  # Perform backpropagation
  loss.backward()

  # Clip gradients: gradients are modified in place
  _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
  _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

  # Adjust model weights
  encoder_optimizer.step()
  decoder_optimizer.step()

  return sum(print_losses) / n_totals

# Full training procedure
def trainIters(model_name, vocab, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):
  # Load batches for each iteration
  training_batches = [batchToTrainData(vocab, [random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]

  # Initializations
  print('Initializing ...')
  start_iteration = 1
  print_loss = 0
  if loadFilename:
    start_iteration = checkpoint['iteration'] + 1

  # Training loop
  print("Training...")
  for iteration in range(start_iteration, n_iteration + 1):
    training_batch = training_batches[iteration - 1]
    # Extract fields from batch
    input_variable, lengths, target_variable, mask, max_target_len = training_batch

    # Run a training iteration with batch
    loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
    print_loss += loss

    # Print progress
    if iteration % print_every == 0:
      print_loss_avg = print_loss / print_every
      print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
      print_loss = 0

    # Save checkpoint
    if (iteration % save_every == 0):
      directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
      if not os.path.exists(directory):
        os.makedirs(directory)
      torch.save({
        'iteration': iteration,
        'en': encoder.state_dict(),
        'de': decoder.state_dict(),
        'en_opt': encoder_optimizer.state_dict(),
        'de_opt': decoder_optimizer.state_dict(),
        'loss': loss,
        'voc_dict': vocab.__dict__,
        'embedding': embedding.state_dict()
      }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


# ------------------------------------------------------------------------------------------------------------

# 
# Evaluation
# 

# Define how we want model to decode encoded input
# Greedy decoding
class GreedySearchDecoder(nn.Module):
  def __init__(self, encoder, decoder):
    super(GreedySearchDecoder, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  def forward(self, input_seq, input_length, max_length):
    # Forward input through encoder model
    encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
    # Prepare encoder's final hidden layer to be first hidden input to the decoder
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # Initialize decoder input with SOS_token
    decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
    # Initialize tensors to append decoded words to
    all_tokens = torch.zeros([0], device=device, dtype=torch.long)
    all_scores = torch.zeros([0], device=device)
    # Iteratively decode one word token at a time
    for _ in range(max_length):
      # Forward pass through decoder
      decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
      # Obtain most likely word token and its softmax score
      decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
      # Record token and score
      all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
      all_scores = torch.cat((all_scores, decoder_scores), dim=0)
      # Prepare current token to be next decoder input (add a dimension)
      decoder_input = torch.unsqueeze(decoder_input, 0)
    # Return collections of word tokens and scores
    return all_tokens, all_scores
    

def preprocessInput(input_str):                                   
  text = input_str.lower()

  # Modify text
  text = text.replace ('"', " ")
  punc = [".", "!", "?", ",", "(", ")", "-", ":", "¡"] # Remove punctuation that messes up regex
  for p in punc:
    text = re.sub("[" + p + "]", " ", text)
  text = re.sub(r"[\t]", "", text)
  text = re.sub(r"['’]", "", text)
  text = re.sub(r"[ ]{2,}", " ", text)
  text = re.sub(r"^[ ]", "", text)
  text = re.sub(r"[\n] ", "", text)
  text = re.sub(r"[ ]$", "", text)
  text = text.replace("\n","")
  return text

def replaceWords(input_str):
  text = input_str
  
  return input_str

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_PAIR_LENGTH):
  ### Format input sentence as a batch
  # words -> indexes
  indexes_batch = [indexesFromSentence(voc, sentence)]
  # Create lengths tensor
  lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
  # Transpose dimensions of batch to match models' expectations
  input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
  # Use appropriate device
  input_batch = input_batch.to(device)
  lengths = lengths.to("cpu")
  # Decode sentence with searcher
  tokens, scores = searcher(input_batch, lengths, max_length)
  # indexes -> words
  decoded_words = [voc.index_to_word[token.item()] for token in tokens]
  return decoded_words

# Most of the manual work was done in this function
def evaluateInput(encoder, decoder, searcher, voc):
  input_sentence = ''
  print("")
  while(1):
    try:
      # Get input sentence
      input_sentence = input('> ')
      # Check if it is quit case
      if input_sentence == 'q' or input_sentence == 'quit': break
      # Normalize sentence
      input_sentence = preprocessInput(input_sentence)
      # Replace words that aren't in vocabulary if possible
      input_sentence = replaceWords(input_sentence)
      # Evaluate sentence
      output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
      # Format and print response sentence
      output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
      print('Kevin:', ' '.join(output_words))

    except KeyError:
      print("Error: Encountered unknown word.")

    
# ------------------------------------------------------------------------------------------------------------

#
# Run model
#

# Configure models
model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
checkpoint_iter = 4000
#loadFilename = None
loadFilename = os.path.join(model_save_dir, model_name, corpus_name,
                '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a ``loadFilename`` is provided
if loadFilename:
  # If loading on same machine the model was trained on
  checkpoint = torch.load(loadFilename)
  # If loading a model trained on GPU to CPU
  #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
  encoder_sd = checkpoint['en']
  decoder_sd = checkpoint['de']
  encoder_optimizer_sd = checkpoint['en_opt']
  decoder_optimizer_sd = checkpoint['de_opt']
  embedding_sd = checkpoint['embedding']
  vocab.__dict__ = checkpoint['voc_dict']

print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(vocab.num_words, hidden_size)
if loadFilename:
  embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, vocab.num_words, decoder_n_layers, dropout)
if loadFilename:
  encoder.load_state_dict(encoder_sd)
  decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
  encoder_optimizer.load_state_dict(encoder_optimizer_sd)
  decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have CUDA, configure CUDA to call
#for state in encoder_optimizer.state.values():
#  for k, v in state.items():
#    if isinstance(v, torch.Tensor):
#      state[k] = v.cuda()

#for state in decoder_optimizer.state.values():
#  for k, v in state.items():
 #   if isinstance(v, torch.Tensor):
#      state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
trainIters(model_name, vocab, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, model_save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename)

# Set dropout layers to ``eval`` mode
encoder.eval()
decoder.eval()

# Initialize search module
searcher = GreedySearchDecoder(encoder, decoder)

# Begin chatting (uncomment and run the following line to begin)
evaluateInput(encoder, decoder, searcher, vocab)