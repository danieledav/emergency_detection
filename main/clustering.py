import numpy as np
import scipy.cluster.hierarchy as hcluster
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import centroid, fcluster
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

cluster = []
clusters = []

cluster.append(vectors[0])
clusters.append(cluster)
tresh = 0.8
centroids = []
centroids.append(vectors[0])
cluster_centroid = 0


#Prendiamo singolo vettore
for x in range(0, len(vectors)):
    max_distance = 0
    for y in range(0, len(centroids)):
        distance = 1 - cosine(vectors[x], centroids[y])
        if (distance > max_distance):
            max_distance = distance
            cluster_centroid = y

    if (max_distance > tresh):
        print("lunghezza cluster prima")
        print(len(clusters))
        cluster.append(vectors[x])
        clusters[cluster_centroid].append(cluster)
        print("lunghezza cluster dopo")
        print(len(clusters))
        print("cluster centroid")
        print(cluster_centroid)

        #calcolare nuovo centroide del cluster
        print(clusters)
        new_centroid = np.mean(cluster[0], axis=0)
        centroids.append(new_centroid)

    else:
        centroids.append(vectors[x])

print(clusters)
print(len(clusters))


