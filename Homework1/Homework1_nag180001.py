import sys
import os
import re
import pickle

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
def process_file(text: str):
  """
  Process given text to extract person data. Return dictionary of persons with IDs as keys.

  Arguments: 
    text: text to process
  """
  
  text = text.split('\n')[1:]
  persons = {}
  for line in text:
    # split on comma to get fields as text variables
    line = line.split(',')

    # modify last name and first name to be capittal case, if necessary
    line[0] = line[0].capitalize()
    line[1] = line[1].capitalize()

    # modify middle initial to be a single upper case letter, if necessary. Use 'X' as middle initial if one is missing
    if (line[2]):
      line[2] = line[2].capitalize()
    else:
      line[2] = 'X'

    # modify id if necessary, using regex.
    # The id should be 2 letters followed by 4 digits.
    # If an id is not in the correct format, output an error message, and allow the user to re-enter a valid ID
    m = re.match('^[\w]{2}[\d]{4}$', line[3])
    while(not m):
      print("ID invalid:", line[3])
      print("ID is two letters followed by 4 digits")
      line[3] = input("Please enter a valid ID: ")
      m = re.match('^[\w]{2}[\d]{4}$', line[3])

    # modify phone number, if necessary, to be in form 999-999-9999. Use regex.
    m = re.match('^([0-9]{3})[-\.)( ]*([0-9]{3})[-\.)( ]*([0-9]{4})$', line[4])
    while ( not m or len(m.groups()) != 3):
      print("Phone", line[4], "is invalid")
      print("Enter phone number in form 123-456-7890")
      line[4] = input("Enter phone number:")
      m = re.match('^([0-9]{3})[-\.)( ]([0-9]{3})[-\.)( ]*([0-9]{4})$', line[4])
    phoneGroups = m.groups()
    line[4] = phoneGroups[0] + "-" + phoneGroups[1] + "-" + phoneGroups[2]

    # Once the data for a person is correct, create a Person object and save the object to a dict of persons, where id is the key.
    # Check for duplicate id and print an error message if an ID is repeated in the input file.
    p = Person(line[0], line[1], line[2], line[3], line[4])
    if p.id in persons.keys():
      print(p.id, "already present, ignoring")
      print()
      continue
    persons[p.id] = p
    print()
  
  # Return the dict of persons to the main function.
  return persons

def main():
  # check sysarg for relative path 'data/data.csv'
  if (len(sys.argv) != 2):
    print('Incorrect number of arguments. Exactly one argument required. Please enter filename.')
    exit()

  # Open file
  filepath = sys.argv[1]
  current_dir: str = os.getcwd()
  with open(os.path.join(current_dir, filepath), 'r') as f:
    text_in = f.read()

  persons = process_file(text_in)

  # Save persons as a pickle file
  pickle.dump(persons, open('persons.p', 'wb')) 

  # Open the pickle file for read, and print each person using the Person display() method to verifiy that the pickle was unpickled correctly.
  persons_test = pickle.load(open('persons.p', 'rb'))  # read binary
  print("Employee list:\n")
  for p in persons_test.values():
    p.display()

if __name__ == '__main__':
  main()



