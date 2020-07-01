
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
import csv

class NeuralNetworkGenre:
    def __init__(self):
        self.loadDataset()
        self.neuralNetwork = MLPClassifier(hidden_layer_sizes=(2, 4), max_iter=2000,learning_rate_init=0.07, activation='identity')

    def loadDataset(self):
        pima = pd.read_csv('API/dataSetGenre.csv', sep=",")
        x = pima.iloc[:, 3:8]
        y = pima.iloc[:, 8]
        standarizacion = StandardScaler().fit_transform(x)
        xStandard = pd.DataFrame(data=standarizacion, columns=x.columns)
        xStandard.head()
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            xStandard, y, test_size=0.2)

    def train(self):
        self.neuralNetwork.fit(self.xTrain, self.yTrain)

    def predict(self, energy,liveness,tempo,speechiness,acousticness):
        xPredict = np.array([energy,liveness,tempo,speechiness,acousticness])
        xPredict = xPredict.reshape((1, 5))
        genre = self.neuralNetwork.predict(xPredict)[0]
        return genre


##neuralNetwork = NeuralNetworkGenre()
##neuralNetwork.train()
##print(neuralNetwork.predict(0.529,0.0856,161.989,0.307,0.0769))
