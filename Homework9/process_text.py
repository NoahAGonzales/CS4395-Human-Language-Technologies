# Process text and write to data/southpark-corpus.txt and data/conversations.txt

import nltk
import opendatasets as od
import pandas as pd
import re
import string
import nltk

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
  text = text.replace ('"', " ")
  punc = [".", "!", "?", ",", "(", ")", "-", ":", "¡"] # Remove punctuation that messes up regex
  for p in punc:
    text = re.sub("[" + p + "]", " ", text)
  text = re.sub(r"o[h]{2,}", "oh", text) # ohs
  text = re.sub(r"a[h]{2,}", "oh", text) # ahs
  text = re.sub(r"[\d]:[\d]{2}", " ", text) # times
  text = re.sub(r"[áä]", "a", text) # character not in ascii
  text = re.sub(r"[ñ]", "n", text) # character not in ascii
  text = re.sub(r"[íî]", "i", text) # character not in ascii
  text = re.sub(r"[ü]", "u", text) # character not in ascii
  text = re.sub(r"[é]", "e", text) # character not in ascii
  text = re.sub(r"[ў]", "y", text) # character not in ascii
  text = re.sub(r"[ôó]", "o", text) # character not in ascii
  text = re.sub(r"[…—™йбн]", " ", text) # replace character not in ascii with a space
  text = re.sub(r"[тщشماچیزیمی‌فروشید؟ت،ثهللهکتنکنعآبخخبа]", "", text) # remove character not in ascii
  text = text.replace(u'\xa0', u' ') # unicode characters, latin space
  text = text.replace(u'\xad', u' ') # unicode characters, soft hyphen
  text = re.sub(r"[\t]", " ", text)
  text = re.sub(r"['’]", "", text)
  text = re.sub(r"[\\]$", " ", text)
  text = re.sub(r"[ ]{2,}", " ", text)
  text = re.sub(r"^[ ]", "", text)
  text = re.sub(r"[\n] ", "", text)
  text = re.sub(r"[ ]$", "", text)
  text = text.replace("\n","")

  # Tokenize to remove stopwords and punctuation
  #tokens = text.split(" ")
  #stop_words = set(stopwords.words('english')) # TODO: try removing stopwords and keeping punctionation
  #tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation] 
  #text = " ".join(tokens)

  return text

# Preprocess text
text = []
for line in df.Line:
  text.append(preprocess_text(line))

# Write conversational text to file
output = open('data/conversations.txt', 'w', encoding='UTF-8')
for i, line in enumerate (text):
  if (i < len(text)-1):
    output.write(line + "\t" + text[i+1] + "\n")
output.close()

text = '\n'.join(text)

# Write corpus to file
output = open('data/southpark-corpus.txt', 'w', encoding='UTF-8')
output.write(text)
output.close()