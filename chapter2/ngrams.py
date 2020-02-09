from nltk.util import ngrams as nltkNgrams
from tokenizers import treebankTokenize as tokenize
def ngrams(s,n):
    # wrapper for nltk ngrams

    # this returns a generator
    tokens = tokenize(s)
    g = nltkNgrams(tokens,n)
    for i in g:
        print(i)
