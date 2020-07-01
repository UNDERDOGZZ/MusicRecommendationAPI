from flask import Flask, jsonify, request, make_response
from neuralNetworkEmotion import NeuralNetworkEmotion
from neuralNetworkGenre import NeuralNetworkGenre
import numpy as np
import csv
import json
from flask_cors import CORS
import pandas as pd
from io import StringIO
import random
app = Flask(__name__)
CORS(app)

@app.route('/recommendatione', methods=['POST'])
def recommendation():
    kinki = float(request.json['kinki'])
    love =float(request.json['love'])
    success = float(request.json['success'])
    hard =float(request.json['hard'])
    need = float(request.json['need'])
    feel = float(request.json['feel'])
    day = float(request.json['day'])
    neww = float(request.json['neww'])
    gone = float(request.json['gone'])
    heart = float(request.json['heart'])
    boy = float(request.json['boy'])
    realli = float(request.json['realli'])
    real = float(request.json['real'])
    think = float(request.json['think'])
    would = float(request.json['would'])
    thing = float(request.json['thing'])
    one = float(request.json['one'])
    hold = float(request.json['hold'])
    sdanger = float(request.json['sdanger'])
    bitch = float(request.json['bitch'])
    leav = float(request.json['leav'])
    danger = float(request.json['danger'])
    around =float(request.json['around'])
    talk = float(request.json['talk'])
    run =float(request.json['run'])
    keep = float(request.json['keep'])
    nigga = float(request.json['nigga'])
    live = float(request.json['live'])
    sweet = float(request.json['sweet'])
    world = float(request.json['world'])
    alway = float(request.json['alway'])
    successt = float(request.json['successt'])
    eye = float(request.json['eye'])
    parti = float(request.json['parti'])
    wait = float(request.json['wait'])
    bodi = float(request.json['bodi'])
    made = float(request.json['made'])
    noth = float(request.json['noth'])
    show = float(request.json['show'])
    bad = float(request.json['bad'])
    turn = float(request.json['turn'])
    longg = float(request.json['longg'])
    home = float(request.json['home'])
    light = float(request.json['light'])
    hand = float(request.json['hand'])
    work = float(request.json['work'])
    nigparti = float(request.json['nigparti'])
    caheart = float(request.json['caheart'])
    walk = float(request.json['walk'])
    

    neurona = NeuralNetworkEmotion()
    neurona.train()
    recommendation = int(neurona.predict(kinki,love,success,hard,need,feel,day,neww,gone,heart,boy,realli,real,think,would,thing,one,hold,sdanger,bitch,leav,danger,around,talk,run,keep,nigga,live,sweet,world,alway,successt,eye,parti,wait,bodi,made,noth,show,bad,turn,longg,home,light,hand,work,nigparti,caheart,walk))
    csv_file = csv.reader(open('API/dataSetEmotion.csv', "r"), delimiter=",")
    possible = []
    line = 0
    for row in csv_file:
        
        if line >=1:
            if int(row[51])==recommendation:
                possible.append(row)
        line+=1
    n = len(possible)
    r = random.randrange(n)
    t = possible[r]

    return jsonify({"title":t[0]})

@app.route('/recommendationg', methods=['POST'])
def recommendationg():
    year = int(request.json['year'])
    energy =float(request.json['energy'])
    liveness = float(request.json['liveness'])
    tempo =float(request.json['tempo'])
    speechiness = float(request.json['speechiness'])
    acousticness = float(request.json['acousticness'])
    
    
    
    
    neurona = NeuralNetworkGenre()
    neurona.train()
    recommendation = int(neurona.predict(energy,liveness,tempo,speechiness,acousticness))
    csv_file = csv.reader(open('API/dataSetGenre.csv', "r"), delimiter=",")
    possible = []
    line = 0
    for row in csv_file:
        
        if line >=1:
            aux = row[2]
            first = aux[len(aux)-2]
            second = aux[len(aux)-1]
            aux = int(first+second)
            if int(row[8])==recommendation and (aux==year or aux-1==year or aux+1==year) :
                possible.append(row)
        line+=1
    n = len(possible)
    r = random.randrange(n)
    t = possible[r]

    return jsonify({"title":t[0]})

@app.route('/track', methods=['POST'])
def dataset():
    song = (request.json['song'])
    datos=pd.read_csv('API/dataSetMusic.csv', sep=",")
    df = pd.DataFrame(datos)
    
    track = (df[df['title']==song]['spotify_link'])
    
    return jsonify({"track":track.to_string(index=False,dtype=False).strip()})

@app.route('/songs')
def songs():
    
    datos=pd.read_csv('API/dataSetEmotion.csv', sep=",")
    df = pd.DataFrame(datos)
    
    return df.to_json(orient='records')

@app.route('/songsg')
def songsg():
    
    datos=pd.read_csv('API/dataSetGenre.csv', sep=",")
    df = pd.DataFrame(datos)
    
    return df.to_json(orient='records')



if __name__ == '__main__':
    app.run(debug=True, port=4000)