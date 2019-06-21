import numpy as np
import scipy.cluster.hierarchy as hcluster
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import centroid, fcluster
from import_csv import importCsv
from tweet2vec import *
from preprocessing import *

# Importing my dataset from the file
data = importCsv('/home/danieledavoli/emergency_detection/Cresci-SWDM15.csv')

# Preprocessing each tweet
for x in range(len(data[1])):

    parsed = doPreprocessing(data[1][x])
    data[1][x] = parsed

print(data[1])

# Rapresentig my tweets as vectors (TF-IDF weight)
vectors = text2tfidf(data[1])

# Test vectors
# vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [8.2, 0, 0], [0, 2, 0]])

# Creating parameter vars and clusters container
clusters = []

tresh = 0.005   # Setting my treshold
cluster_centroid = 0
centroids = np.array([vectors[0]])
cluster = np.array([vectors[0]])
clusters.append(cluster)

# Printing init parameter values
print("Init cluster:")
print(cluster)
print("Init clusters:")
print(clusters)
print("Init centroids:")
print(centroids)

# For each vector
for x in range(1, len(vectors)):
    if (x%100 == 0):
        print(x)

    max_distance = 0
    cluster_centroid = 0

    # Computing distance of the vector from each centroid
    for y in range(0, len(centroids)):
        distance = 0
        distance = 1 - cosine(vectors[x], centroids[y])

        # Getting the max distance
        if (distance > max_distance):
            max_distance = distance
            cluster_centroid = y

    # Assigning a vector to a cluster if enough close and recomputing the new cluster centroid
    if (max_distance > tresh):
        cluster = clusters[cluster_centroid]
        cluster = np.vstack((cluster, vectors[x]))
        clusters[cluster_centroid] = cluster

        # Recomputing the new cluster centroid
        new_centroid = np.mean(cluster, dtype=np.float64, axis=0)
        centroids[cluster_centroid] = new_centroid

    # Creating a new cluster
    else:
        centroids = np.vstack((centroids, vectors[x]))
        cluster = np.array([vectors[x]])
        clusters.append(cluster)


# Printing my centroids
print("Array di centroidi")
print(centroids)

# Printing the number of my clusters
print(clusters)
print(len(clusters))


