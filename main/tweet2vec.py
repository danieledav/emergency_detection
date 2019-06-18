from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

#List of text documents
text = ["Tornare in camera e trovare l'armadio aperto",
        "Altra scossa forte di terremoto. #terremoto",
        "e dire che non ci tenevo a fare la notte bianca, #terremoto"]

#Create the vocabulary

vec = CountVectorizer()
vec.fit(text)

print(vec.vocabulary_)

#Trasform a document as a vector in the vocabulary space

vector = vec.transform(text)


print(vector.shape)
print(type(vector))
print(vector.toarray())

#TF-IDF version
ifidf = TfidfVectorizer()
ifidf.fit(text)

print(ifidf.vocabulary_)
print(ifidf.idf_)

vector2 = ifidf.transform(text)

print(vector2.shape)
print(vector2.toarray())

