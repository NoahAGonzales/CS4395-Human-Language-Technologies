# CS4395-Human-Language-Technologies
Natural Language Processing Portfolio

## Overview of NLP
This pdf document describes an overview of natural language processing and also describes my personal interest in the topic.

The document can be found [here](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/18a6401e5b0cacc9d8d76f95f4aaa12601676591/Overview_of_NLP.pdf).

## Homework 1 - Text Processing with Python
This python program processes a csv data file by extracting and validating information about the persons in such.

### To run
Execute the following command in /Homework1:
> python Homework1_nag180001.py *path.csv*

Homework1_nag180001.py can be found [here](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/03f8d9fa7e4305c78397c615cd8d086383db9301/Homework1/Homework1_nag180001.py).

### Takeaway
Python is easy to use for text procesing. 
The many built-in functions and regex support make it easy to tokenize text.
However, the lack of built-in support for typing makes validating input more difficult - not that type support can't be added.

This assignment was a good refresher on python.
I had previously worked with python in other classes, such as AI and Software Engineering, but I do not frequently work with it outside of an academic setting.
Consequently, I knew the concepts for parsing strings and regex - pickle was new to me but not as involved.

## Homework 2 - Word Guess Game
This python program is a word guessing game that utilizes NLTK features to process text for the game.

### To run
Execute the following command in /Homework2:
> python Homework2_nag180001.py *path.txt*

Homework2_nag180001.py can be found [here](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/21e50defca4c63080bb6f54d82e7bb5567324c86/Homework2/Homework2_nag180001.py).


## Homework 3 - Wordnet
This is a notebook containing some play with WordNet.
The notebook can be found [here](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/129340376f09cbc09028c70dad6e6191b16bf686/Homework3/HLT3_nag180001.ipynb)


## Homework 4 - N Gram Language Model
This assignment consists of 3 parts: a [python script](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/b73221efc7121e7ad271c3bd6b00f95a8a226477/Homework4/Homework4-1_nag180001.py) that writes dictionaries of unigrams and bigrams for each language (English, French, and Italian) to files, a [python script](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/b73221efc7121e7ad271c3bd6b00f95a8a226477/Homework4/Homework4-2_nag180001.py) that classifies a test file of text as one of those three languages and determines the accuracy of the classification, and a [paper](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/b73221efc7121e7ad271c3bd6b00f95a8a226477/Homework4/Homework4-3_nag180001.pdf) containing information on n-grams and their use.

### To run
To run the first script, which writes dictionaries of unigrams and bigrams to files, execute the following command in /Homework4:
> python Homework4-1_nag180001.py
Ensure that the necessary files (data folder) are /Homework4.


To run the second script, which classifies the test file as one of three languages, execute the following command in /Homework4:
> python Homework4-2_nag180001.py
Ensure that the necessary files (data folder) are /Homework4.

## Homework 5 - Sentence parsing
This assignment involves parsing a sentence in three different ways, including constituent parsing, dependency parsing, and semantic role label parsing and analyzing the pros and cons of each method. [This](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/362e742b2f1b7cda775642c4b2a0e1db9499bbbd/Homework5/Homework5_nag180001.pdf) document contains each parse and the pros and cons of such.

## Homework 6 - Finding or Building a Corpus / web_crawler
This assignment was to create a corpus using a webcrawler.
The webcrawler scapes text off of a starting site and related sites linked from that starting site.
The text is then used to create a knowledge base, which in this case is a pickled dictionary containing terms as keys and an array of sentences containing each term as the value. [This](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/bf0a1971c0296f8f68b95bf14c40c47c171a0443/Homework6/Homework6-webcrawler_nag180001.pdf) document describes the process and assignment.
The python file can be found [here](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/bf0a1971c0296f8f68b95bf14c40c47c171a0443/Homework6/Homework6_nag180001.py)

## Homework 7 - Text Classification 1
This assignment was to classify text using three different methods: Naive-Bayes, Regression, and Neural Networks. 
The performance of the three methods was analyzed.
The text classification can be found [here](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/1d57aca76b518aab90e381ddcb94a47b48eafc5e/Homework7/Homework7_nag180001.pdf).

## Homework 8 - ACL Paper Summary
This assignment was to write a summary of a reseach paper concerning natural language processing. The summary document can be found [here](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/blob/87b0f845f25be33d938b36fd84b68a22ba36160e/Homework8/Homework8_nag180001.pdf)

## Homework 9 - Chatbot
[This](https://github.com/NoahAGonzales/CS4395-Human-Language-Technologies/tree/main/Homework9) assignment was to create a chatbot. I did not like the concept of creating a chatbot that is a simple interface for querying information. Therefore, my goal for the chatbot was to create a personality that is fun to interact with instead of informative. I think that I acomplished this goal: I created a bot that talks like characters from the TV show "South Park". However, there were some compromises. The chatbot is built on a RNN made with PyTorch. User profiles and incorporating such into the dialog was not able to be accomplished. I still think that this project was fun and successful. 
