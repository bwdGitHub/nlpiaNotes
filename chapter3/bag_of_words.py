from collections import Counter
from collections import OrderedDict
import copy
import numpy as np

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

    def cosine_similarity(self,i,j):
        # simple cosine similarity between document i and j
        v = dict2array(self.tfVectorByDict(i))
        w = dict2array(self.tfVectorByDict(j))
        return cosine(v,w)


def dict2array(d):
    # helper for mapping dictionary values to np arrays
    return np.array(list(d.values()))

def cosine(v,w):
    # compute the dot product of v/|v| and w/|w|
    # i.e. cos(Angle(v,w))
    return np.dot(v/np.linalg.norm(v),w/np.linalg.norm(w))
