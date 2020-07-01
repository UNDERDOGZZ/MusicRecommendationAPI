from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import csv

neuralNetwork = MLPClassifier(hidden_layer_sizes=(2, 4), max_iter=2000,
                              learning_rate_init=0.07, activation='identity')

pima = pd.read_csv('creatingDataSet/bagOfWords.csv', sep=",", nrows=49)
x = pima.iloc[:, 2:51]
y = pima.iloc[:, 51]
standarizacion = StandardScaler().fit_transform(x)
xStandard = pd.DataFrame(data=standarizacion, columns=x.columns)
xStandard.head()
xTrain, xTest, yTrain, yTest = train_test_split(xStandard, y, test_size=0.01)
neuralNetwork.fit(xTrain, yTrain)

pima2 = pd.read_csv('creatingDataSet/bagOfWords.csv', sep=",", skiprows=49)
xpredict = pima2.iloc[:, 2:51]
ypredict = pima2.iloc[:, 51]
standarizacion = StandardScaler().fit_transform(xpredict)
xStandardPredict = pd.DataFrame(data=standarizacion, columns=xpredict.columns)
xStandardPredict.head()

df = pd.read_csv("creatingDataSet/bagOfWords.csv")

for i in range(len(xStandardPredict)):
    auxPredict = xStandardPredict.iloc[i]
    auxPredict = auxPredict.values.reshape(1, -1)
    df.at[i+49, "Emotion"] = neuralNetwork.predict(auxPredict)[0]

df.to_csv("creatingDataSet/dataSet.csv", index=False)
