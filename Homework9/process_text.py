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
  text = text.replace ('"', " ")
  punc = [".", "!", "?", ",", "(", ")", "-"] # Remove punctuation that messes up regex
  for p in punc:
    text = re.sub("[" + p + "]", " ", text)
  text = re.sub(r"o[h]{2,}", "oh", text)
  text = re.sub(r"a[h]{2,}", "oh", text)
  text = re.sub(r"[\t]", " ", text)

  # Tokenize to remove stopwords and punctuation
  tokens = word_tokenize(text)
  stop_words = set(stopwords.words('english')) # TODO: try removing stopwords and keeping punctionation
  tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation] 

  return text


# Preprocess text
text = []
for line in df.Line:
  text.append(preprocess_text(line))

# Write conversational text to output file
output = open('data/conversations.txt', 'w', encoding='UTF-8')
for i, line in enumerate (text):
  if (i < len(text)-1):
    output.write(line + "\t" + text[i+1] + "\n")
output.close()

text = '\n'.join(text)

output = open('data/corpus.txt', 'w', encoding='UTF-8')
output.write(text)
output.close()