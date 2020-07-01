from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv

textFile = open("creatingBagOfWords/corpus.txt", "r")
corpus = textFile.read()

tokens = word_tokenize(corpus, "english")
tokens = [word.lower() for word in tokens if word.isalpha()]

textFile = open("creatingBagOfWords/tokens1.txt", "w")
n = textFile.write(str(tokens))
textFile.close()
