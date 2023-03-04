import os
import nltk
import pickle
import math
from nltk import word_tokenize
from nltk.util import ngrams
nltk.download("punkt")

def read_lines(file_path):
  current_dir: str = os.getcwd()
  with open(os.path.join(current_dir, file_path), 'r') as f:
    return f.readlines()

def compute_prob (text, unigram_dict, bigram_dict, num_tokens, vocab_size):
  unigrams = word_tokenize(text)
  bigrams = list(ngrams (unigrams, 2))

  p_gt = 1 # Good-Turing
  p_laplace = 1 # laplace smoothing
  p_log = 0

  for bigram in bigrams:
    n = bigram_dict[bigram] if bigram in bigram_dict else 0
    n_gt = bigram_dict[bigram] if bigram in bigram_dict else 1/num_tokens
    d = unigram_dict[bigram[0]] if bigram [0] in unigram_dict else 0
    if d == 0:
      p_gt = p_gt * (1/num_tokens)
    else:
      p_gt = p_gt * (n_gt / d)
    p_laplace = p_laplace * ((n + 1) / (d + vocab_size))
    p_log = p_log + math.log((n + 1) / (d + vocab_size))

  # 30% accuracy with simplified Good-Turing smoothing
  # 97.3% Accuracy with laplace smoothing
  # 99.3% accuracy with log of laplace smoothing
  return p_log

def main():
  # load unigrams and bigrams
  english_u = pickle.load(open("english1.p", "rb"))
  english_b = pickle.load(open("english2.p", "rb"))
  french_u = pickle.load(open("french1.p", "rb"))
  french_b = pickle.load(open("french2.p", "rb"))
  italian_u = pickle.load(open("italian1.p", "rb"))
  italian_b = pickle.load(open("italian2.p", "rb"))
  vocab_size = len(english_u.items()) + len(french_u.items()) + len(italian_u.items())

  # Open test file
  lines = read_lines("data/LangId.test")
  line_num = 1

  # open file to write classifications
  c = open("classifications.txt", "w")

  # Find probabilities
  for line in lines:
    line = line.replace('\n', '')
    ps = {
      "English": compute_prob(line, english_u, english_b, len(word_tokenize(line)), vocab_size),
      "French": compute_prob(line, french_u, french_b, len(word_tokenize(line)), vocab_size),
      "Italian": compute_prob(line, italian_u, italian_b, len(word_tokenize(line)), vocab_size),
    }
    highest_p = [l for l, p in sorted(ps.items(), key=lambda x: x[1], reverse=True)][0]
    c.write(str(line_num) + " " + highest_p + "\n")
    line_num += 1

  c.close()

  # Calculate accuracy
  c_lines = read_lines("classifications.txt")
  s_lines = read_lines("data/LangId.sol")

  num_correct = 0
  for line_num in range(0,len(c_lines)):
    if c_lines[line_num] == s_lines[line_num]:
      num_correct += 1

  print("Accuracy: ", num_correct / 300)

if __name__ == '__main__':
  main()

    
    