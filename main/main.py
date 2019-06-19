from import_csv import importCsv
from tweet2vec import *
from preprocessing import *

data = importCsv('/home/danieledavoli/emergency_detection/Cresci-SWDM15.csv')

for x in range(len(data[1])):

    parsed = doPreprocessing(data[1][x])
    data[1][x] = parsed

print(data[1])

text2tfidf(data[1])