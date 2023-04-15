# Chatbot
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import string

def preprocess_text(text):    
  # Tokenize the text into individual words
  tokens = word_tokenize(text)

  # Lowercase all words
  tokens = [word.lower() for word in tokens]

  # Remove stopwords and punctuation
  stop_words = set(stopwords.words('english')) # TODO: try removing stopwords and keeping punctionation
  tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation] 
  tokens = [token.replace("...","") for token in tokens]

  # Lemmatize words
  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(token) for token in tokens]

  # Return the filtered tokens as a string
  return " ".join(tokens)


def main():
  characters = [ 'guts','casca','griffith' ]

  # Process text
  for character in characters:
    f = open("data/" + character + "/unformatted.txt", "r", encoding="utf-8")
    output = open("data/" + character + "/formatted.txt", "w", encoding="utf-8")

    unprocessed_text = f.read()
    output.write(preprocess_text(unprocessed_text))
    

  return
  
if __name__=="__main__":
  main()