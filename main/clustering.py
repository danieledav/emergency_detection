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
#vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [8.2, 0, 0], [0, 2, 0]])

print(vectors)
print("vectors[0]:")
print(vectors[0])

tresh = 0.015
centroids = np.array([vectors[0]])
cluster = np.array([vectors[0]])
clusters = []

clusters.append(cluster)
vec = np.array([])


print("Init cluster:")
print(cluster)
print("Init clusters:")
print(clusters)

cluster_centroid = 0

print("Init centroids:")
print(centroids)

#Prendiamo singolo vettore

for x in range(1, len(vectors)):
    if (x%100 == 0):
        print(x)

    max_distance = 0
    cluster_centroid = 0


    for y in range(0, len(centroids)):
        distance = 0
        distance = 1 - cosine(vectors[x], centroids[y])
        #print("distance:")
        #print(distance)

        if (distance > max_distance):
            max_distance = distance
            cluster_centroid = y
            #print("custer_centroid:")
            #print(cluster_centroid)

    #print("max distance:")
    #print(max_distance)
    if (max_distance > tresh):
        #print("Clusters prima:")
        #print(clusters)
        #print("vectors[x]:")
        #print(vectors[x])
        #print("Cluster centroid:")
        #print(cluster_centroid)
        cluster = clusters[cluster_centroid]
        #print("copia cluster preso in considerazione:")
        #print(cluster)
        cluster = np.vstack((cluster, vectors[x]))
        clusters[cluster_centroid] = cluster
        #clusters.insert(cluster_centroid, cluster)
        #print("cluster dopo:")
        #print(cluster)
        #print("cluster centroid")
        #print(cluster_centroid)

        #calcolare nuovo centroide del cluster
        #print("Clusters dopo:")
        #print(clusters)

        new_centroid = np.mean(cluster, dtype=np.float64, axis=0)
        #print("New centroid")
        #print(new_centroid)
        #print(cluster_centroid)
        centroids[cluster_centroid] = new_centroid
        #centroids = np.delete(centroids, cluster_centroid)
        #np.insert(centroids, cluster_centroid, new_centroid)
        #centroids = np.vstack((centroids, new_centroid))



    else:

        centroids = np.vstack((centroids, vectors[x]))
        cluster = np.array([vectors[x]])
        clusters.append(cluster)



print("Array di centroidi")
print(centroids)

print(clusters)
print(len(clusters))


