import nltk
import opendatasets as od
import pandas as pd
import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Dataset
# {"username":"noahagonzo","key":"b7e4d1aedc1148c648f8fcef1ab58905"}
od.download("https://www.kaggle.com/datasets/tovarischsukhov/southparklines")

# Import training data
df = pd.read_csv('southparklines/All-seasons.csv', header=0, encoding='utf-8')

print(df.shape)
print(df.head())

# Define a function for preprocessing text to remove special characters and alter format of expressive words
def preprocess_text(text):
  text = text.lower()

  # Modify text
  text = text.replace("\n","")
  punc = [".", "!", "?", ","] # Remove punctuation that messes up regex
  for p in punc:
    text = re.sub("[" + p + "]", " ", text)
  text = re.sub(r"o[h]{2,}", "oh", text)
  text = re.sub(r"a[h]{2,}", "oh", text)

  # Tokenize to remove stopwords and punctuation
  tokens = word_tokenize(text)
  stop_words = set(stopwords.words('english')) # TODO: try removing stopwords and keeping punctionation
  tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation] 

  return text


# Preprocess text
text = []

for line in df.Line:
  text.append(preprocess_text(line))

text = '\n'.join(text)

output = open('data/corpus.txt', 'w', encoding='UTF-8')
output.write(text)