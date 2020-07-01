from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

ownStopWords = ["got", "nah", "na", "ca", "way",
                "ai", "want", "wan", "see", "get", "gets", "even", "oh", "ya", "yeah", "woah", "like", "go", "make", "take", "fuck", "hm", "mhm", "mmm", "let", "tell", "know", "ta", "back",
                "never", "like", "hey", "ooh", "come", "say", "gon", "could", "wo", "said",
                "ever", "every", "put", "out", "still", "good"]

textFile = open("creatingBagOfWords/tokens2.txt", "r", encoding='latin-1')
tokens2 = textFile.read().split(",")

for i in range(len(tokens2)):
    tokens2[i] = tokens2[i].replace(" ", "")
    tokens2[i] = tokens2[i].replace("'", "")
    tokens2[i] = tokens2[i].replace("[", "")
    tokens2[i] = tokens2[i].replace("]", "")

cont = 0
cleanTokens = tokens2[:]
for token in tokens2:
    print(cont)
    cont += 1
    if token in ownStopWords:
        cleanTokens.remove(token)

textFile = open("creatingBagOfWords/tokens3.txt", "w")
n = textFile.write(str(cleanTokens))
textFile.close()
