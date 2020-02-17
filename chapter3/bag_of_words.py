from collections import Counter
from collections import OrderedDict
import copy
import numpy as np
import math

class bag_of_words:
    # a very simple bag of words implementation

    def __init__(self,documents):
        # assume documents is a list of list of strs
        self.bag = []
        # risk - set is unordered
        # the book later uses an OrderedDict as the zero vector representation
        self.lexicon = set()
        self.lens = []

        for document in documents:
            self.bag.append(Counter(document))
            self.lens.append(len(document))
            for token in document:
                self.lexicon.add(token)

        self.zero = OrderedDict((token,0) for token in self.lexicon)

    def tfPerDoc(self,word,i):
        return self.bag[i][word]/self.lens[i]       

    def tf(self,word):
        tf = []
        for i in range(len(self.bag)):
            tf.append(self.tfPerDoc(word,i))
        return tf
    
    def tfVector(self,i):
        # return tf vector for document i
        v = []
        for token in self.lexicon:
            v.append(self.tfPerDoc(token,i))
        return v

    def tfVectorByDict(self,i):
        # implementation via the ordered dict representation
        # p78
        v = copy.copy(self.zero)
        for token in self.bag[i]:
            v[token] = self.tfPerDoc(token,i)
        return v

    def docFreq(self):
        v = copy.copy(self.zero)
        for key in v:
            counter = 0
            for bag in self.bag:
                if key in bag:
                    counter+=1
            v[key] = counter
        numDocs = len(self.bag)
        for key in v:
            v[key] /= numDocs
        return v

    def idf(self,word):
        return 1/self.docFreq()[word]

    def logIdf(self,word):
        return math.log(self.idf(word))

    def tfidf(self,word,i):
        bag = self.bag[i]
        v = self.tfVectorByDict(i)
        df = self.docFreq()
        if word not in df or df[word]<1e-10:
            return 0.0
        else:
            return v[word]/df[word]

    def logTfIdf(self,word,i):
        bag = self.bag[i]
        v = self.tfVectorByDict(i)
        logIdf = self.logIdf(word)
        return v[word]/logIdf

    def tfidfVector(self,i):
        v = copy.copy(self.zero)
        tf = self.tfVectorByDict(i)
        df = self.docFreq()
        bag = self.bag[i]
        for word in bag:
            v[word] = tf[word]/df[word]
        return v

    def cosine_similarity(self,i,j):
        # simple cosine similarity between document i and j
        v = dict2array(self.tfVectorByDict(i))
        w = dict2array(self.tfVectorByDict(j))
        return cosine(v,w)

    def doc2tfidf(self,query):
        # map a new document to a tfidf vector
        w = copy.copy(self.zero)
        counts = Counter(query)
        for k,v in counts.items():
            freq = 0
            for bag in self.bag:
                if k in bag:
                    freq+=1
            if freq == 0:
                continue
            tf = v/len(query)
            idf = len(self.bag)/freq
            w[k] = tf*idf
        return w

    def cosine_similarity_query(self,i,query):
        v = dict2array(self.tfVectorByDict(i))
        w = dict2array(self.doc2tfidf(query))
        return cosine(v,w)

def dict2array(d):
    # helper for mapping dictionary values to np arrays
    return np.array(list(d.values()))

def cosine(v,w):
    # compute the dot product of v/|v| and w/|w|
    # i.e. cos(Angle(v,w))
    return np.dot(v/np.linalg.norm(v),w/np.linalg.norm(w))
