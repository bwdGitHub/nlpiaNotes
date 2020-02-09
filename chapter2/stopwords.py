# the book actually suggests stopword removal is not all too useful on large vocabularies, especially when n-gram features are required.
import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stop_words
def removeStopwords(stopwords,document):
    return [token for token in document if token not in stopwords]

def removeStopwordsNLTK(document):
    # wrapper for nltk stopword removal
    stopwords = nltk.corpus.stopwords.words('english')
    return removeStopwords(stopwords,document)

def removeStopwordsSklearn(document):
    # wrapper for sklearn stopwords
    stopwords = sklearn_stop_words
    return removeStopwords(stopwords,document)

def removeStopwordsNLTKSklearn(document):
    # dumb wrapper for just showing the union method
    stopwords = sklearn_stop_words.union(nltk.corpus.stopwords.words('english'))
    return removeStopwords(stopwords,document)