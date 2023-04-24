# The text has a lot of characters from a different language that had to be cut out in preprocessing
# This script writes lines with offending characters to data/unicode.txt

import re

regexp = re.compile(r'[^\x00-\xff]')

with open('data/southpark-corpus.txt', 'r', encoding = 'utf-8') as f:
  with open('data/testing/unicode.txt', 'w', encoding = 'utf-8') as output:
    lines = f.readlines()
    for line in lines:
      if regexp.search(line):
        output.write(line)