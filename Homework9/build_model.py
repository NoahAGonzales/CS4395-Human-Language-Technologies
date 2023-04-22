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

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# ------------------------------------------------------------------------------------------------------------

#
# Preprocess data
#

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

# Create vocabulary and vectorize
vocab = set()
num_words = 3
word_to_index = {}
word_to_count = {}
index_to_word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}

# Load the data
print('Opening text to build vocab...')
with open('data/corpus.txt', 'r', encoding='UTF-8') as f:
  text = f.read()
  tokens = nltk.word_tokenize(text)
  print('Building vocabulary...')
  for token in tokens:
    if token not in word_to_index:
      num_words+=1
      word_to_index[token] = num_words
      word_to_count[token] = 1
      index_to_word[num_words] = token
      num_words += 1
    else:
      word_to_count[token] += 1

# Get Pairs
MAX_PAIR_LENGTH = 15
with open('data/conversations.txt', 'r', encoding='UTF-8') as f:
  lines = f.readlines()
  pairs = [[re.sub(r"[ ]{2,}", " ", re.sub(r"[\n]", "", s)) for s in l.split('\t')] for l in lines]
  # Filter pairs to only those that have a token length less than MAX_PAIR_LENGTH
  pairs = [pair for pair in pairs if len(pair[0].split(' ')) < MAX_PAIR_LENGTH and len(pair[1].split(' ')) < MAX_PAIR_LENGTH]

print( pairs[:10] )

# Trim less used words - NOT going to do :)



# ------------------------------------------------------------------------------------------------------------


#
# Prepare data for model
#

# ------------------------------------------------------------------------------------------------------------

#
# Define model
#

# ------------------------------------------------------------------------------------------------------------

#
# Define training procedure
#

# ------------------------------------------------------------------------------------------------------------

# 
# Evaluation
# 

# ------------------------------------------------------------------------------------------------------------

#
# Save model
#

