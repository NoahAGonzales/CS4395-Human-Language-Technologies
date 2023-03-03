import sys
import os
import nltk
import pickle
from nltk import word_tokenize
from nltk.util import ngrams
nltk.download("punkt")

def process(filename):
    #Open file
    current_dir: str = os.getcwd()
    with open(os.path.join(current_dir, filename), 'r', encoding='UTF-8') as f:
        text_in = f.read()

    # remove newlines
    text_in = text_in.replace("\n", "")

    # tokenize
    unigrams = word_tokenize(text_in)

    # bigrams list
    bigrams = list(ngrams(unigrams,2))

    unigram_dict = {u: unigrams.count(u) for u in set (unigrams)}
    bigram_dict = {b: bigrams.count(b) for b in set (bigrams)}

    return (unigram_dict, bigram_dict)

def main():
    # Open file
    english = process("data\LangId.train.English")
    french = process("data\LangId.train.French")
    italian = process("data\LangId.train.Italian")
    
    # pickle unigrams and bigrams
    pickle.dump(english[0], open('english1.p', 'wb'))
    pickle.dump(english[1], open('english2.p', 'wb'))
    pickle.dump(french[0], open('french1.p', 'wb'))
    pickle.dump(french[1], open('french2.p', 'wb'))
    pickle.dump(italian[0], open('italian1.p', 'wb'))
    pickle.dump(italian[1], open('italian2.p', 'wb'))

  

if __name__ == '__main__':
  main()

    
    