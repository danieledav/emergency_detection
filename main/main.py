from import_csv import importCsv
from tweet2vec import text2vector, text2tfidf

text = importCsv('/home/danieledavoli/emergency_detection/Cresci-SWDM15.csv')

print(text)

text2vector(text)

text2tfidf(text)