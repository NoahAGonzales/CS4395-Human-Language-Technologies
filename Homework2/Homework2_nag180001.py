import sys
import os
import re
import datetime
import calendar
from random import seed
from random import randint
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#nltk.download('all')

class Person():
  """
  This class holds data for a person

  Attributes:
    last: last name of the person
    first: first name of the person
    mi: middle initial of the person
    id: id of the person
    phone: phone number of the person
  """
  last = ''
  first = ''
  mi = ''
  id = ''
  phone = ''

  def __init__(self, last, first, mi, id, phone):
    self.last = last
    self.first = first
    self.mi = mi
    self.id = id
    self.phone = phone

  # Display person
  def display(self): 
    print('\t', 'ID: ', self.id)
    print('\t', self.first, self.mi, self.last)
    print('\t', self.phone)
    print()

# Read in the file
def preprocess(text: str):
  """
  Process given text. Returns tokens (all) and nouns

  Arguments: 
    text: text to process
  """
  
  # Make raw text lowercase
  text = text.lower()
  # Tokenize
  tokens = word_tokenize(text)
  # Reduce to tokens that are alpha
  tokens = [token for token in tokens if re.match('^[^\W\d]*$',token)]
  # Remove stopwords
  stop_words = set(stopwords.words('english'))
  tokens = [token for token in tokens if not token in stop_words]
  # Enforce min length > 5
  tokens = [token for token in tokens if len(token) > 5]

  # Lemmatize tokens
  wnl = WordNetLemmatizer()
  lemmas = set([wnl.lemmatize(t) for t in tokens])

  # POS
  tags = nltk.pos_tag(lemmas)

  # nouns
  nouns = [t[0] for t in tags if t[1][0] == 'N']

  print("Number of tokens:", len(tokens))
  print("Number of nouns:", len(nouns))

  return (tokens, nouns)

def guessing_game(nouns):
  score = 5
  guess = ''
  history = []

  while score >= 0 and guess != '!':

    # randomly choose a word
    seed(int(calendar.timegm(datetime.datetime.now().timetuple())))
    word = nouns[randint(0, 50)]
    # Start game
    guess = ''
    guesses = ['_' for letter in word]
    history = []

    while score >= 0 and guess != '!':
      # Print current guesses
      s = ''
      for letter in guesses:
        s += letter + " "
      print()
      print(s)
      
      # Ask for letter
      guess = input("Enter letter: ").lower()

      # Check for a singular letter
      if len(guess) != 1:
        print("Guesses must be letters")
      # Check if already guessed
      elif guess in history:
        print("You have already guessed the following letters:", ",".join(str(x) for x in history))
      # Evaluate guess
      elif guess in word:
        score += 1
        history.append(guess)
        print("Right! Score is ", score)
        indices = [i for i in range(len(word)) if word.startswith(guess, i)]
        for i in indices:
          guesses[i] = guess
        # Guessed entire word
        if "_" not in guesses:
          print()
          print(" ".join(str(x) for x in guesses))
          print("You solved it!\n")
          print("Current score: ", score, "\n")
          print("Guess another word!\n")
          break
      # Guess wrong
      elif guess != '!':
        score -= 1
        history.append(guess)
        print("Sorry, guess again. Score is ", score)

  print("Final score is ", score)  

def main():
  # check sysarg for relative path 'data/data.csv'
  if (len(sys.argv) != 2):
    print('Incorrect number of arguments. Exactly one argument required. Please enter filename.')
    exit()

  # Open file
  filepath = sys.argv[1]
  current_dir: str = os.getcwd()
  print(os.path.join(current_dir, filepath))

  with open(os.path.join(current_dir, filepath), 'r', encoding='utf-8') as f:
    text_in = f.read()

  # Tokenize
  tokens = word_tokenize(text_in)

  # Find lexical diversity
  lex_diversity = len(set(tokens)) / len(tokens)
  print('Lexical diversity:', '{:.2f}'.format(lex_diversity))

  # Preprocess the text
  t = preprocess(text_in)
  tokens = t[0]
  nouns = t[1]

  # Dictionary of nouns and their count and get the top 50 most frequent
  freqDict = {}
  for noun in nouns:
    freqDict[noun] = tokens.count(noun)
  mostFreq = [noun[0] for noun in sorted(freqDict.items(), key=lambda x: x[1], reverse=True)[:50]]
  print(mostFreq)

  # Play guessing game
  guessing_game(mostFreq)
  
  
if __name__ == '__main__':
  main()



