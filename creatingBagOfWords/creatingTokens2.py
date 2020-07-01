from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

textFile = open("creatingBagOfWords/tokens1.txt", "r")
tokens = textFile.read().split(",")

cleanTokens = tokens[:]
for token in tokens:
    if token in stopwords.words('english'):
        cleanTokens.remove(token)

textFile = open("creatingBagOfWords/tokens2.txt", "w")
n = textFile.write(str(cleanTokens))
textFile.close()
