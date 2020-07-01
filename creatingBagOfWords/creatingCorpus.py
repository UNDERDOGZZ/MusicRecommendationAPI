from bs4 import BeautifulSoup
import csv

corpus = ""
with open('lyrics.csv', 'r', errors="ignore") as file:
    line = 1
    reader = csv.reader(file)
    songs = []
    for row in reader:
        if line != 0 and line < 5000:
            if line != 1 and row[29] != "unknown":
                corpus = corpus + row[30]
            line += 1
        else:
            break

textFile = open("creatingBagOfWords/corpus.txt", "w")
n = textFile.write(str(corpus))
textFile.close()
