import numpy as np
def encode(corpus,document):
    # encode a document in the one-hot vectors corresponding to the vocabulary of corpus
    # both inputs are assumed to be strings
    vocab = getVocab(corpus)    
    return encodeDocument(vocab,document)

def getVocab(s):
    tokens = tokenize(s)
    return sorted(set(tokens))

def tokenize(s):
    # naive tokenizer
    return str.split(s)

def encodeDocument(voc,doc):
    tokens = tokenize(doc)
    # add an "out of vocabulary" element
    arr = initArray(len(tokens),len(voc)+1)
    for i,tok in enumerate(tokens):
        if tok in voc:
            arr[i,voc.index(tok)]=1
        else:
            arr[i,len(voc)]=1
    return arr

def initArray(n,m):
    return np.zeros((n,m),int)
