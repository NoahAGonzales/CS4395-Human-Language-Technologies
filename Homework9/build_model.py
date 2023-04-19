import nltk
import torch
import torch.nn as nn
nltk.download('punkt')

# Load the data
print('Opening corpus...')
with open('data/corpus.txt', 'r', encoding='utf8') as f:
  text = f.read()


# TODO: Rewrite this

class LSTMGenerator(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super(LSTMGenerator, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x, hidden):
    out, hidden = self.lstm(x, hidden)
    out = self.fc(out)
    return out, hidden

  def init_hidden(self, batch_size):
    return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
            torch.zeros(self.num_layers, batch_size, self.hidden_size))

# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Sequence text
print('Sequencing...')
seq_length = 10
sequences = []
for i in range(len(tokens) - seq_length):
  seq = tokens[i:i+seq_length]
  target = tokens[i+seq_length]
  sequences.append((seq, target))


print(sequences[:10])

quit()
exit()



# TODO: rewrite these names

# Model parameters
input_size = len(vocab)
hidden_size = 256
output_size = len(vocab)
num_layers = 2
batch_size = 128
learning_rate = 0.01
num_epochs = 100

# Initialize the model and optimizer
model = LSTMGenerator(input_size, hidden_size, output_size, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()








# Save the model
torch.save(model.state_dict(), "model.pth")