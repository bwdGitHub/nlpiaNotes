import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

#general notes - normalization and vocab compression always comes at a cost, or a trade
# more recall for less precision
# lower memory for less expressivity
# for IR/search the loss of precision for high recall can be mitigated by ranking algorithms

def naiveCaseNormalization(document):
    # simple normalization
    return [token.lower() for token in document]

def slightlyLessNaiveNormalization(document):
    # only normalize the first tokens in a sentence
    # really this needs a sentence-end detector.
    # also proper nouns should be detected up front and preserved.
    for i in range(len(document)-1):
        if i==0 or document[i-1]==".":
            document[i] = document[i].lower()
    return document

def sStemmer(document):
    # stem plurals and 's
    return [re.findall('^(.*ss|.*?)(s)?$',token)[0][0].strip("'") for token in document]

def nltkStemmer(document):
    # wrapper for nltk Porter stemmer.
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in document]

def lemmatize(document):
    #wrapper for nltk wordnet lemmatizer
    # this does better when each token in document has a pos tag.
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in document]