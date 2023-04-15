# Chatbot
import nltk
from nltk import word_tokenize
nltk.download('stopwords')
STOPWORDS = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import string


def preprocess_text(text):    
  # Tokenize the text into individual words
  tokens = word_tokenize(text.lower())
  # Remove stopwords and punctuation
  stop_words = set(stopwords.words('english') + list(string.punctuation)) # TODO: try removing stopwords and keeping punctionation
  tokens1 = [token for token in tokens if token not in stop_words] 
  # Return the filtered tokens as a string
  return ' '.join(tokens1)

# Define a function to generate a chatbot response
def generate_response(user_input):
  # Preprocess and tokenize the user input
  preprocessed_input = preprocess_text(user_input)
  input_vector = vectorizer.transform([preprocessed_input])
  # Use the classifier to predict a response
  predicted_category = clf.predict(input_vector)[0]
  # Choose a random movie review from the predicted category
  reviews_in_category = movie_reviews.fileids(predicted_category)
  review_id = random.choice(reviews_in_category)
  review_text = movie_reviews.raw(review_id)
  # Return the review text as the chatbot response
  return review_text


def main():
  # import text
  f = open("data/guts/0.txt" + "r", encoding="utf-8")
  unprocessed_text = f.read()

  # Extract Features
  vectorizer = CountVectorizer()
  corpus = [unprocessed_text]
  vectorizer.fit_transform([preprocess_text(text) for text in corpus])


  # temp
  # Train ML model
  corpus = [(preprocess_text(movie_reviews.raw(fileid)), category)
          for category in movie_reviews.categories()]


  return
  
if __name__=="__main__":
  main()