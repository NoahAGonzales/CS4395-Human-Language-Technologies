# Chatbot
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
from nltk.stem import WordNetLemmatizer


def preprocess_sentence(sent):    
  # Tokenize sentence
  words = nltk.word_tokenize(sent)
  # Remove stop words
  words = [w for w in words if w not in STOPWORDS]
  # Lemmatize words
  lemmatizer = WordNetLemmatizer()
  words = [lemmatizer.lemmatize(w) for w in words]
  return ' '.join(words)

def classify(sen):
  sentence = preprocess_sentence(sen)
  score = {}
  for c in classes:
      score[c] = 0
  # Calculate score for each output class
  for word in nltk.word_tokenize(sentence):
      if word in corpus_words:
          for c in classes:
              score[c] += (1 + class_words[word][c]) / (len(classes) + corpus_words[word])
  # Return the class with highest score
  return max(score, key=score.get)


# Define the interface
while True:
  # Get user input
  sentence = input("You: ")
  # Get chatbot response
  response = classify(sentence)
  # Print chatbot response
  print("Chatbot: " + response)