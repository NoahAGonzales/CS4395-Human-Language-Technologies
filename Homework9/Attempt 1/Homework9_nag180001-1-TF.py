import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import string

# TODO: Data has to have labels with it

max_length = 100
trunc_type='post'
padding_type='post'

def preprocess_text(text):    
  # Tokenize the text into individual words
  tokens = word_tokenize(text)

  # Lowercase all words
  tokens = [word.lower() for word in tokens]

  # Remove stopwords and punctuation
  stop_words = set(stopwords.words('english')) # TODO: try removing stopwords and keeping punctionation
  tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation] 

  # Lemmatize words
  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(token) for token in tokens]

  # Return the filtered tokens as a string
  return " ".join(tokens)

# Define function to predict answer
def predict_answer(model, tokenizer, question):
  # Preprocess question
  question = preprocess_text(question)
  # Convert question to sequence
  sequence = tokenizer.texts_to_sequences([question])
  # Pad sequence
  padded_sequence = pad_sequences(sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
  # Predict answer
  pred = model.predict(padded_sequence)[0]
  # Get index of highest probability
  idx = np.argmax(pred)
  # Get answer
  answer = tokenizer.index_word[idx]
  return answer


def main():
  # Import data
  f = open("data/" + "guts" + "/formatted.txt", "r", encoding="utf-8")
  data = f.read()

  # Set parameters 
  # TODO: Adjust naming
  vocab_size = 3200
  embedding_dim = 64
  oov_tok = "<OOV>"
  training_size = len(data)

  # Create tokenizer
  # TODO: Adjust naming
  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
  tokenizer.fit_on_texts(data)
  word_index = tokenizer.word_index

  # Create sequences
  # TODO: Adjust naming
  sequences = tokenizer.texts_to_sequences(data)
  padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

  # Create training data
  # TODO: Adjust naming
  training_data = padded_sequences[:training_size]
  training_labels = padded_sequences[:training_size]

  # Build model
  # TODO: Adjust naming

  model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
  ])
  
  #model = tf.keras.Sequential([
  #  tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
  #  tf.keras.layers.Dropout(0.2),
  #  tf.keras.layers.Conv1D(64, 5, activation='relu'),
  #  tf.keras.layers.MaxPooling1D(pool_size=4),
  #  tf.keras.layers.LSTM(64),
  #  tf.keras.layers.Dense(64, activation='relu'),
  #  tf.keras.layers.Dense(vocab_size, activation='softmax')
  #])

  # Compile model
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Train model
  num_epochs = 10
  history = model.fit(training_data, training_labels, epochs=num_epochs, verbose=2)

  # Start chatbot
  while True:
    question = input('You: ')
    answer = predict_answer(model, tokenizer, question)
    print('Chatbot:', answer)

  return
  
if __name__=="__main__":
  main()
