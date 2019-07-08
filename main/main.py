from import_csv import importCsv
from tweet2vec import *

from preprocessing import doPreprocessing
from tweet2vec import text2tfidf
from clustering import AgglomerativeClustering

from classification import *

# Importing my dataset from the file
data = importCsv('/home/danieledavoli/emergency_detection/Cresci-SWDM15.csv')

# Preprocessing each tweet
for x in range(len(data[1])):

    parsed = doPreprocessing(data[1][x])
    data[1][x] = parsed

print(data[1])


# Rapresentig my tweets as vectors (TF-IDF weight)
vectors = text2tfidf(data[1])
data.append(vectors)


# Training my agglomerative clustering algorithm and check the numbers of cluster I have with a specific threshold
n_clusters1 = AgglomerativeClustering(vectors, tresh=0.005)
n_clusters2 = AgglomerativeClustering(vectors, tresh=0.01)
n_clusters3 = AgglomerativeClustering(vectors, tresh=0.1)
n_clusters4 = AgglomerativeClustering(vectors, tresh=0.5)
n_clusters5 = AgglomerativeClustering(vectors, tresh=0.95)

print("Treshold: 0.005\t" + str(n_clusters1))
print("Treshold: 0.01\t" + str(n_clusters2))
print("Treshold: 0.1\t" + str(n_clusters3))
print("Treshold: 0.5\t" + str(n_clusters4))
print("Treshold: 0.95\t" + str(n_clusters5))



dataset = BuildNewStructureData(vectors, data)

# Training my SVM algorithm first with gamma=1 then with gamma=100
#SVMalgorithmKind(dataset, 0.05)
#SVMalgorithmKind(dataset, 0.09)
#SVMalgorithmKind(dataset, 1)
#SVMalgorithmRelevantNotRelevant(dataset, 500)


# Training my Naive Bayes algorithm
BayesalgorithmKind(dataset)
BayesalgorithmRelevantNotRelevant(dataset)
