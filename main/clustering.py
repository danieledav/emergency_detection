import numpy
import scipy.cluster.hierarchy as hcluster
from import_csv import importCsv
from tweet2vec import *
from preprocessing import *

data = importCsv('/home/danieledavoli/emergency_detection/Cresci-SWDM15.csv')

for x in range(len(data[1])):

    parsed = doPreprocessing(data[1][x])
    data[1][x] = parsed

print(data[1])

vectors = text2tfidf(data[1])

print(vectors)

#Prendiamo singolo vettore
for 
#lista in cui memorizzo i centroidi

#calcolare la distanza cosin fra vettore e i centroidi

#memorizzo il valore massimo di cosin similarity

#assegnare il vettore al centroide
