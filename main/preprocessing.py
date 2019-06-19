import nltk
import string
import re
from nltk import pos_tag
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

'''
the prepocessing is composed by:
	- punctuaction elimination
	- stop words elimination
	- stemming
	- digits elimination
	- extra spaces elimination
'''



#main function
def doPreprocessing(tweet_message):

	tweet_message = removePunct(tweet_message)

	tweet_message = removeStopWords(tweet_message)

	tweet_message = doStemming(tweet_message)

	tweet_message = removeDigits(tweet_message)

	#remove double spaces and unwanted spaces
	return re.sub(' +',' ',tweet_message).strip()

#divide the string in tokens while eliminating the punctuation
def removePunct(tweet_message):
	tokenizer = RegexpTokenizer(r'\w+')
	return tokenizer.tokenize(tweet_message.lower())

#remove stop word in the tokenized tweet message
def removeStopWords(tokens):
	stop_words = set(stopwords.words('italian')) #english
	filtered_sentence = []
	for token in tokens:
		if token not in stop_words:
			filtered_sentence.append(token)
	return filtered_sentence

def doStemming(tokens):
	ita_stemmer = nltk.stem.snowball.ItalianStemmer()
	for i in range(0, len(tokens)):
		tokens[i] = ita_stemmer.stem(tokens[i])
	tweet_message = " ".join(str(x) for x in tokens)
	return tweet_message

def removeDigits(tweet_message):
	remove_digits = str.maketrans('', '', string.digits)
	clean_message = tweet_message.translate(remove_digits)
	return clean_message

