# the book rants on and on that LSA is SVD.
# In particular - encode your corpus in a matrix W which is (vocab_size,corpus_size).
# SVD this into p topics.
# encode = bag_of_words, tfidf, whatever.

# W -> USV' 
# U - left singular, (vocab_size,num_topics) -> U is the cross-correlation of words and topics.
from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models,prettify_tdm
import numpy as np
import pandas as pd

def catdog_svd():
    # get the term document matrix - tdm
    bow_svd = lsa_models()
    tdm = bow_svd[0]['tdm']
    U,S,Vt = np.linalg.svd(tdm)
    U = pd.DataFrame(U,index = tdm.index).round(2)
    return U

