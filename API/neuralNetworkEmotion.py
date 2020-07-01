
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
import csv
import random

class NeuralNetworkEmotion:
    def __init__(self):
        self.loadDataset()
        self.neuralNetwork = MLPClassifier(hidden_layer_sizes=(2, 4), max_iter=2000,learning_rate_init=0.07, activation='identity')

    def loadDataset(self):
        pima = pd.read_csv('API/dataSetEmotion.csv', sep=",")
        x = pima.iloc[:, 2:51]
        y = pima.iloc[:, 51]
        standarizacion = StandardScaler().fit_transform(x)
        xStandard = pd.DataFrame(data=standarizacion, columns=x.columns)
        xStandard.head()
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            xStandard, y, test_size=0.2)

    def train(self):
        self.neuralNetwork.fit(self.xTrain, self.yTrain)

    def predict(self, kinki,love,success,hard,need,feel,day,neww,gone,heart,boy,realli,real,think,would,thing,one,hold,sdanger,bitch,leav,danger,around,talk,run,keep,nigga,live,sweet,world,alway,successt,eye,parti,wait,bodi,made,noth,show,bad,turn,longg,home,light,hand,work,nigparti,caheart,walk):
        xPredict = np.array([kinki,love,success,hard,need,feel,day,neww,gone,heart,boy,realli,real,think,would,thing,one,hold,sdanger,bitch,leav,danger,around,talk,run,keep,nigga,live,sweet,world,alway,successt,eye,parti,wait,bodi,made,noth,show,bad,turn,longg,home,light,hand,work,nigparti,caheart,walk])
        xPredict = xPredict.reshape((1, 49))
        emotion = self.neuralNetwork.predict(xPredict)[0]
        return emotion


##neuralNetwork = neuralNetwork()
##neuralNetwork.train()
##print(neuralNetwork.predict(10,4,1,0,0,1,0,0,0,2,0,0,0,1,1,2,1,1,2,2,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0))

#datos=pd.read_csv('API/dataSetMusic.csv', sep=",")
#df = pd.DataFrame(datos)
#df.to_string
#artista = df[df['title']=='All Girls Are The Same']['spotify_link']
#print (artista)
