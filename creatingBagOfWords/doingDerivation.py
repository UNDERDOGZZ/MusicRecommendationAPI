from nltk.stem import SnowballStemmer
import nltk

textFile = open("creatingBagOfWords/tokens4.txt", "r", encoding='latin-1')
tokens4 = textFile.read().split(",")

for i in range(len(tokens4)):
    tokens4[i] = tokens4[i].replace(" ", "")
    tokens4[i] = tokens4[i].replace("'", "")
    tokens4[i] = tokens4[i].replace("[", "")
    tokens4[i] = tokens4[i].replace("]", "")

stemmer = SnowballStemmer('english')
tokens5 = [stemmer.stem(t) for t in tokens4]

ownStopWords = ["big", "mind", "suck", "crazy", "everyth", "gave",
                "better", "cool", "play", "boo", "time", "dunk", "drench", "free", "la", "right", "ay", "stay", "stop", "damn", "told", "face", "look", "man", "find", "place", "give", "fli", "tri", "littl", "left", "away"]

cont = 0
cleanTokens5 = tokens5[:]
for token in tokens5:
    print(cont)
    cont += 1
    if token in ownStopWords:
        cleanTokens5.remove(token)

mostFreqTokens = []
frecuencias = nltk.FreqDist(cleanTokens5)
for t, f in frecuencias.items():
    if f > 1500:
        mostFreqTokens.append(t)

textFile = open("creatingBagOfWords/tokens5.txt", "w")
n = textFile.write(str(mostFreqTokens))
textFile.close()
