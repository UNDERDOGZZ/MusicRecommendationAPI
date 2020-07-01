from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

textFile = open("creatingBagOfWords/tokens3.txt", "r", encoding='latin-1')
tokens3 = textFile.read().split(",")

for i in range(len(tokens3)):
    tokens3[i] = tokens3[i].replace(" ", "")
    tokens3[i] = tokens3[i].replace("'", "")
    tokens3[i] = tokens3[i].replace("[", "")
    tokens3[i] = tokens3[i].replace("]", "")

equivalence = ["butterflies", "sweetheart", "feelings", "lovegame", "husband", "madonna",
               "showty", "shorty", "fiance", "heart", "crush", "chick", "love", "baby", "girl", "bae", "we"]
rootEquivalence = "love"

equivalence2 = ["billionare", "millionare", "pavements", "magazine", "maserati", "hustling", "foreign", "bitches", "streets", "picasso", "hustler", "versace", "clothes", "million",
                "swagin", "stacks", "commas", "winner", "hustla", "money", "gucci", "shiny", "racks", "spend", "livin", "drive", "broke", "bens", "benz", "whip", "cars", "drip", "hoes", "swag",
                "rari", "rich", "flex", "star", "car", "mil", "fall"]
rootEquivalence2 = "success"

equivalence3 = ["disappointing", "heartbreaker", "disappointed", "expectation", "heartbreak", "pretending", "solitude", "jealousy", "jealous", "heart", "alone",
                "call", "exes", "us", "ex"]
rootEquivalence3 = "heart"

equivalence4 = ["misunderstood", "bittersweet", "goosebumps", "addiction", "addicted", "feelings", "sadness", "misread", "sadness", "crying", "sorrow", "regret",
                "stuck", "agony", "call", "cuts", "numb", "void", "sad", "cry"]
rootEquivalence4 = "feelings"

equivalence5 = ["backwood", "partying", "partiyin", "alcohol", "tonight", "cocaine", "bottles", "dance", "booze", "drugs", "percs", "xanny", "joint", "club",
                "xans", "boof", "weed", "kush", "lean", "boof", "high", "lsd", "gas", "og", "night"]
rootEquivalence5 = "party"

equivalence6 = ["seduction", "slippery", "lollipop", "cocaine", "coochie", "sensual", "bedroom", "kissing", "naughty", "mouths", "slutty", "exotic", "tottie", "kinky",
                "lumps", "thong", "brain", "xanny", "freak", "jeans", "thicc", "thick", "boobs", "head", "slut", "lean", "sexy", "kiss", "thot", "lips", "neck", "sex", "cum", "wet",
                "ass", "bed"]
rootEquivalence6 = "kinky"

equivalence7 = ["hardest", "fistful", "villian", "niggas", "bodied", "bullet", "fight", "cruel", "bully", "ak-47", "smoke", "beam", "guns", "boys", "slap", "fire",
                "lava", "heat", "hood", "gang", "uzi", "gun", "hit", "die"]
rootEquivalence7 = "danger"

equivalence8 = ["compliment", "butterfly", "brighter", "kindness", "sweetest", "lullaby", "unicorn", "smiles", "joyful", "summer", "spring", "sweet", "happy",
                "honey", "child", "joy", "friend", "life", "hope", "life"]
rootEquivalence8 = "sweet"

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

textFile = open("creatingBagOfWords/tokens4.txt", "w")
n = textFile.write(str(tokens11))
textFile.close()
