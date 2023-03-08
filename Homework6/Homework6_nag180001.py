import requests
from bs4 import BeautifulSoup


def web_crawler(url, history):
    urls = []

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    for a in soup.select('a'):
        print(a)


    return urls

def scrape_text():
    # Write text to file

    return None

def clean_text():
    # Clean text
    
    return None

def extract_terms():
    return None


def main():
    web_crawler('https://onepiece.fandom.com/wiki/Portgas_D._Ace/History', [])

if __name__ == '__main__':
  main()