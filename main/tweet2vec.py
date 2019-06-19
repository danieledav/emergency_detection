from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from import_csv import importCsv

# List of text documents
#text = ["Tornare in camera e trovare l'armadio aperto",
#        "Altra scossa forte di terremoto. #terremoto",
     #   "e dire che non ci tenevo a fare la notte bianca, #terremoto"]

# text = importCsv('/home/danieledavoli/emergency_detection/Cresci-SWDM15.csv')


def text2vector(text_p):

        # Create the vocabulary

        vec = CountVectorizer()
        vec.fit(text_p)

        print(vec.vocabulary_)

        # Trasform a document as a vector in the vocabulary space

        vector = vec.transform(text_p)

        print(vector.shape)
        print(type(vector))
        print(vector.toarray())


def text2tfidf(text_p):

        # TF-IDF version
        ifidf = TfidfVectorizer()
        ifidf.fit(text_p)

        print(ifidf.vocabulary_)

        vector = ifidf.transform(text_p)

        return vector.toarray()





