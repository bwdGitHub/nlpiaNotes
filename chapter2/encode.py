import numpy as np
import pandas as pd
def encode(corpus,document):
    # encode a document in the one-hot vectors corresponding to the vocabulary of corpus
    # both inputs are assumed to be strings
    vocab = getVocab(corpus)
    onehot = encodeDocument(vocab,document)
    df = pd.DataFrame(onehot,columns=vocab+['out_of_vocab'])
    return df

def getVocab(s):
    tokens = tokenize(s)
    return sorted(set(tokens))

def tokenize(s):
    # naive tokenizer
    return str.split(s)

def lowMemBinaryEncodeDocument(voc,doc):
    # only store the frequency info of tokens in doc using a dict.
    # this is clearly less memory intense than storing the vectors
    enc = {}
    tokens = tokenize(doc)
    for tok in tokens:
        # use voc as the set of tokens you care about
        # alternative is to collect all of them and decide a vocab after the fact.
        if tok in voc:
            enc[tok] = 1
        else:
            enc['out_out_vocab']=1
    return enc

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

def sentence2dict(sentence):
    return dict([(token,1) for token in tokenize(sentence)])

def encodeAsBinaryDataFrame(sentence,i):
    # do everything on one line, because why not, it's not like this was intended to be read.
    # copied this from the book.
    #
    # this is really just a long way to say "give me a dataframe with columns named via the tokens and 1 in every position."
    df = pd.DataFrame(pd.Series(sentence2dict(sentence)),columns=['sent{}'.format(i)]).T
    return df

def encodeCorpusAsBinaryDataFrame(corpus):
    sentences = corpus.split('\n')
    enc = {}
    for i,sent in enumerate(sentences):
        enc['sent{}'.format(i)] = sentence2dict(sent)
    df = pd.DataFrame.from_records(enc).fillna(0).astype(int).T
    return df

def initArray(n,m):
    return np.zeros((n,m),int)
