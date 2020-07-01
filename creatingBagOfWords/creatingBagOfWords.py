from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import csv

textFile = open("creatingBagOfWords/tokens5.txt", "r", encoding='latin-1')
tokens5 = textFile.read().split(",")
textFile.close()

for i in range(len(tokens5)):
    tokens5[i] = tokens5[i].replace(" ", "")
    tokens5[i] = tokens5[i].replace("'", "")
    tokens5[i] = tokens5[i].replace("[", "")
    tokens5[i] = tokens5[i].replace("]", "")

with open('lyrics.csv', 'r', errors="ignore") as file:
    line = 1
    reader = csv.reader(file)
    songs = []
    idOfSong = 1
    for row in reader:
        if line != 0 and line < 5000:
            if line != 1 and row[29] != "unknown":
                song = []
                songTitle = row[2]
                artist = row[4]
                lyrics = row[30]
                song.append(idOfSong)
                song.append(songTitle)
                song.append(artist)
                song.append(lyrics)
                songs.append(song)
                idOfSong += 1
            line += 1
        else:
            break


def convertLyricsToTokens(lyrics):
    ownStopWords = ["got", "nah", "na", "ca", "way",
                    "ai", "want", "wan", "see", "get", "gets", "even", "oh", "ya", "yeah", "woah", "like", "go", "make", "take", "fuck", "hm", "mhm", "mmm", "let", "tell", "know", "ta", "back",
                    "never", "like", "hey", "ooh", "come", "say", "gon", "could", "wo", "said",
                    "ever", "every", "put", "out", "still", "good", "big", "mind", "suck", "crazy", "everyth", "gave",
                    "better", "cool", "play", "boo", "time", "dunk", "drench", "free", "la", "right", "ay", "stay", "stop", "damn", "told",
                    "face", "look", "man", "find", "place", "give", "fli", "tri", "littl", "left", "away"]

    equivalence = ["butterflies", "sweetheart", "feelings", "lovegame", "husband", "madonna",
                   "showty", "shorty", "fiance", "heart", "crush", "chick", "love", "baby", "girl", "bae", "we"]
    rootEquivalence = "love"
    equivalence2 = ["billionare", "millionare", "pavements", "magazine", "maserati", "hustling", "foreign", "bitches", "streets", "picasso", "hustler", "versace", "clothes", "million",
                    "swagin", "stacks", "commas", "winner", "hustla", "money", "gucci", "shiny", "racks", "spend", "livin", "drive", "broke", "bens", "benz", "whip", "cars", "drip", "hoes", "swag",
                    "rari", "rich", "flex", "star", "car", "mil"]
    rootEquivalence2 = "success"
    equivalence3 = ["disappointing", "heartbreaker", "disappointed", "expectation", "heartbreak", "pretending", "solitude", "jealousy", "jealous", "heart", "alone",
                    "call", "exes", "us", "ex"]
    rootEquivalence3 = "heart"
    equivalence4 = ["misunderstood", "bittersweet", "goosebumps", "addiction", "addicted", "feelings", "sadness", "misread", "sadness", "crying", "sorrow", "regret",
                    "stuck", "agony", "call", "cuts", "numb", "void", "sad", "cry"]
    rootEquivalence4 = "feelings"
    equivalence5 = ["backwood", "partying", "partiyin", "alcohol", "tonight", "cocaine", "bottles", "dance", "booze", "drugs", "percs", "xanny", "joint", "club",
                    "xans", "boof", "weed", "kush", "lean", "boof", "high", "lsd", "gas", "og"]
    rootEquivalence5 = "party"
    equivalence6 = ["seduction", "slippery", "lollipop", "cocaine", "coochie", "sensual", "bedroom", "kissing", "naughty", "mouths", "slutty", "exotic", "tottie", "kinky",
                    "lumps", "thong", "brain", "xanny", "freak", "jeans", "thicc", "thick", "boobs", "head", "slut", "lean", "sexy", "kiss", "thot", "lips", "neck", "sex", "cum", "wet",
                    "ass", "bed"]
    rootEquivalence6 = "kinky"
    equivalence7 = ["hardest", "fistful", "villian", "niggas", "bodied", "bullet", "fight", "cruel", "bully", "ak-47", "smoke", "beam", "guns", "boys", "slap", "fire",
                    "lava", "heat", "hood", "gang", "uzi", "gun", "hit"]
    rootEquivalence7 = "danger"
    equivalence8 = ["compliment", "butterfly", "brighter", "kindness", "sweetest", "lullaby", "unicorn", "smiles", "joyful", "summer", "spring", "sweet", "happy",
                    "honey", "child", "joy"]
    rootEquivalence8 = "sweet"

    tokens = word_tokenize(lyrics, "english")
    tokens = [word.lower() for word in tokens if word.isalpha()]

    tokens2 = tokens[:]
    for token in tokens:
        if token in stopwords.words('english'):
            tokens2.remove(token)

    tokens3 = tokens2[:]
    for token in tokens2:
        if token in ownStopWords:
            tokens3.remove(token)

    tokens4 = tokens3[:]
    for sin in equivalence:
        tokens4 = [w.lower().replace(sin, rootEquivalence) for w in tokens4]

    tokens5 = tokens4[:]
    for sin in equivalence2:
        tokens5 = [w.lower().replace(sin, rootEquivalence2) for w in tokens5]

    tokens6 = tokens5[:]
    for sin in equivalence3:
        tokens6 = [w.lower().replace(sin, rootEquivalence3) for w in tokens6]

    tokens7 = tokens6[:]
    for sin in equivalence4:
        tokens7 = [w.lower().replace(sin, rootEquivalence4) for w in tokens7]

    tokens8 = tokens7[:]
    for sin in equivalence5:
        tokens8 = [w.lower().replace(sin, rootEquivalence5) for w in tokens8]

    tokens9 = tokens8[:]
    for sin in equivalence6:
        tokens9 = [w.lower().replace(sin, rootEquivalence6) for w in tokens9]

    tokens10 = tokens9[:]
    for sin in equivalence7:
        tokens10 = [w.lower().replace(sin, rootEquivalence7) for w in tokens10]

    tokens11 = tokens10[:]
    for sin in equivalence8:
        tokens11 = [w.lower().replace(sin, rootEquivalence8) for w in tokens11]

    stemmer = SnowballStemmer('english')
    tokens12 = [stemmer.stem(t) for t in tokens11]

    return tokens12


with open("creatingBagOfWords/bagOfWords.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Title", "Artist", "kinki", "love", "success", "hard", "need", "feel", "day", "new", "gone", "heart", "boy", "realli", "real", "think",
                     "would", "thing", "one", "hold", "sdanger", "bitch", "leav", "danger", "around", "talk", "run", "keep", "nigga", "live", "sweet", "world", "alway", "successt", "eye",
                     "parti", "wait", "bodi", "made", "noth", "show", "bad", "turn", "long", "home", "light", "hand", "work", "nigparti", "caheart", "walk", "Emotion"])
    cont = 0
    for song in songs:
        dictAux = {}
        tokensAux = convertLyricsToTokens(song[3])
        for i in range(len(tokens5)):
            times = 0
            for j in range(len(tokensAux)):
                if tokensAux[j] == tokens5[i]:
                    times += 1
            dictAux.update({tokens5[i]: times})
        writer.writerow([song[1], song[2], dictAux.get("kinki"), dictAux.get("love"), dictAux.get("success"), dictAux.get("hard"), dictAux.get("need"), dictAux.get("feel"),
                         dictAux.get("day"), dictAux.get("new"), dictAux.get("gone"), dictAux.get(
            "heart"), dictAux.get("boy"), dictAux.get("realli"), dictAux.get("real"),
            dictAux.get("think"), dictAux.get("would"), dictAux.get("thing"), dictAux.get(
            "one"), dictAux.get("hold"), dictAux.get("sdanger"), dictAux.get("bitch"),
            dictAux.get("leav"), dictAux.get("danger"), dictAux.get("around"), dictAux.get(
            "talk"), dictAux.get("run"), dictAux.get("keep"), dictAux.get("nigga"),
            dictAux.get("live"), dictAux.get("sweet"), dictAux.get("world"), dictAux.get(
            "alway"), dictAux.get("successt"), dictAux.get("eye"), dictAux.get("parti"),
            dictAux.get("wait"), dictAux.get("bodi"), dictAux.get("made"), dictAux.get(
            "noth"), dictAux.get("show"), dictAux.get("bad"), dictAux.get("turn"),
            dictAux.get("long"), dictAux.get("home"), dictAux.get("light"), dictAux.get(
            "hand"), dictAux.get("work"), dictAux.get("nigparti"), dictAux.get("caheart"),
            dictAux.get("walk"), "c"])
        print(cont)
        cont += 1
