import requests
from bs4 import BeautifulSoup
import re
import math
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')

def web_crawler(url, history):
  page = requests.get(url)
  soup = BeautifulSoup(page.content, 'html.parser')
  regex = (
          "^((https:\/\/)|(http:\/\/))" +
          "(?!.*(https:\/\/)|.*(http:\/\/))" + # Double links?
          "(?!ru|ja|ar|be|br|ca|cs|bg|de|el|es|fa|fr|gl|ko|it|he|lb|lt|nl|sv|mk|pl|pt|sr|sh|th|tr|vi|zh|uk)" + # Other countries
          "(?!foundation|donate|wikimediafoundation|commons|www\.wikidata|.*action=edit)" + # wiki links that are useless
          "(?!sport24|grand|natalie|www\.asahi|.*ddnavi)" + # misc
          "(?!.*\.jp|.*\.fr|.*\.au)" # domains
          )
  external_urls = [url.attrs['href'] for url in soup.find_all('a', href = re.compile(regex))][:10]

  # Check for bad urls
  remove = []
  for id, external_url in enumerate(external_urls):
    try:
      print("Trying", id, external_url)
      page = requests.get(external_url)
    except: 
      print("ERROR", id, external_url)
      remove.append(external_url)

  for r in remove:
    external_urls.remove(r)

  return external_urls

def scrape_text(url, id):
  # Open file to write text
  f = open("output/unformatted/text_" + str(id) + ".txt", "w", encoding="utf-8")

  # Get text and write to file
  try:
    page = requests.get(url)
  except:
    print(id, " link errored out")
    return False
  soup = BeautifulSoup(page.content, 'html.parser')
  for p in soup:#.select('p'):
    f.write(p.get_text())

  # Clean up
  f.close()

  return True

def clean_text(id):
  # Open text file
  f_unformatted = open("output/unformatted/text_" + str(id) + ".txt", "r", encoding="utf-8")
  unformatted = f_unformatted.readlines()

  # Open filters
  to_remove = open("filters/remove.txt", "r", encoding="utf-8").readlines()
  to_remove_line = open("filters/remove_line.txt", "r", encoding="utf-8").readlines()

  
  # Apply filters to lines

  # remove lines containing terms to remove
  r_l = []
  for line in unformatted:
    for r in to_remove_line:
      if line.lower().find(r.lower()) != -1:
        r_l.append(line)
        break

  for r in r_l:
    unformatted.remove(r)

  # Remove empty lines
  while '' in unformatted:
    unformatted.remove('')

  unformatted = "\n".join(unformatted)
  
  # Apply filters to raw text
  for remove in to_remove:
    unformatted = unformatted.replace(remove, ' ')

  # Modify text


  # Last step: remove white space
  unformatted = unformatted.replace('\n',' ')
  #unformatted = unformatted.replace()
  #unformatted = unformatted.replace('\t',' ')

  # Open file to write cleaned up text to
  f_formatted = open("output/formatted/text_" + str(id) + ".txt", "w", encoding="utf-8")

  # Clean text
  f_formatted.write(unformatted)

  # Clean up
  f_unformatted.close()
  f_formatted.close()

  
  return None

def create_tf_dict(text):
  tf_dict = {}
  tokens = word_tokenize(text)
  tokens = [t for t in tokens if t not in STOPWORDS and t.isalpha()]


  # get term frequencies
  for t in tokens:
      if t in tf_dict:
          tf_dict[t] += 1
      else:
          tf_dict[t] = 1

  # Set term frequencies
  #token_set = set(tokens)
  #tf_dict = {t:tokens.count(t) for t in token_set}

   # normalize tf by number of tokens
  for t in tf_dict.keys():
    tf_dict[t] = tf_dict[t] / len(tokens)
      
  return tf_dict

def extract_terms(urls):
  # tf
  vocab = set()
  tf_dicts = []
  for id, url in enumerate(urls):
    # Get and preprocess text
    f = open("output/formatted/text_" + str(id) + ".txt", 'r', encoding="utf-8")
    text = f.read().lower()
    f.close()
    text = text.replace('\n', ' ')
    dict = create_tf_dict(text)
    tf_dicts.append(dict)
    # Add to vocab
    if id == 0:
      vocab = set(dict.keys())
    else:
      vocab.union(set(dict.keys()))

  # idf
  idf_dict = {}
  vocab_by_topic = [x.keys() for x in tf_dicts]
  for term in vocab:
    temp = ['x' for voc in vocab_by_topic if term in voc]
    idf_dict[term] = math.log((1+len(urls)) / (1+len(temp)))

  print("Number of unique words:", len(vocab))

  # tf-idf
  tf_idf_dicts = []
  for tf_dict in tf_dicts:
    tf_idf_dict = {}
    #for t in tf_dict.keys():
      #tf_idf_dict[t] = tf_dict[t] * idf_dict[t]
    #tf_idf_dicts.append(tf_idf_dict)
  

  #doc_term_weights = sorted(tf_idf_dicts[0].items(), key=lambda x:x[1])
  #print(doc_term_weights[:5])


  return None


def main():
  # Crawl web
  external_urls = web_crawler('https://en.wikipedia.org/wiki/Berserk_%28manga%29', [])
  print("Web crawled!", len(external_urls), "urls found")

  # Extract text
  for id, external_url in enumerate(external_urls):
    scrape_text(external_url, id)

  # Clean up text
  for id, external_url in enumerate(external_urls):
    clean_text(id)

  # Extract terms
  extract_terms(external_urls)

if __name__ == '__main__':
  main()