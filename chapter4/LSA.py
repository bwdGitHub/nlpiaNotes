# the book rants on and on that LSA is SVD.
# In particular - encode your corpus in a matrix W which is (vocab_size,corpus_size).
# SVD this into p topics.
# encode = bag_of_words, tfidf, whatever.

# W -> USV' 
# U - left singular, (vocab_size,num_topics) -> U is the cross-correlation of words and topics.
from nlpia.book.examples.ch04_catdog_lsa_sorted import lsa_models,prettify_tdm # note - have to debug this, usually adding encoding='utf8' in places.
import numpy as np
import pandas as pd

def catdog_svd():
    # get the term document matrix - tdm
    bow_svd = lsa_models()
    tdm = bow_svd[0]['tdm']
    U,S,Vt = np.linalg.svd(tdm)
    # U is vocab_size x latent_dim (num_topics)
    U = pd.DataFrame(U,index = tdm.index).round(2)
    # S actually just returns the diagonal to save space.
    # It can be reshaped to a diagonal latent_dim x latent_dim matrix, or a latent_dim x corpus_size matrix.
    S = np.diag(S) # use fill_diagonal with a non-square zeros matrix to get latent_dim x corpus_size.
    zeros = np.zeros((S.shape[0],Vt.shape[0]-S.shape[1]))
    S = np.concatenate((S,zeros),axis=1)
    # Trick: Make S into an identity to get rid of variance information
    S[S>0] = 1

    return (U,S,Vt)

# Since the important singular values are already aranged to the left of S
# You can compress U into topics by removing columns on the right.
# It helps to have a metric for how much information is lost

def reconstruction_error(dims_to_drop,tdm):
    U,s,Vt = np.linalg.svd(tdm)
    for dim_to_drop in range(len(s),len(s)-dims_to_drop,-1):
        s[dim_to_drop-1] = 0    
    S_approx = np.diag(s)
    zeros = np.zeros((S_approx.shape[0],Vt.shape[0]-S_approx.shape[1]))
    S_approx = np.concatenate((S_approx,zeros),axis=1)
    reconstruction = U.dot(S_approx).dot(Vt)
    e = mse(tdm.values,reconstruction)
    return e

def mse(A,B):
    # protip: use as many parentheses as possible to look smart
    # don't commit errors, no one will know
    return np.sqrt((((A-B).flatten()**2).sum())/np.product(A.shape))

# plot dims_to_drop against reconstruction_error to set how this decreases.
# since this is svd, some dimensions can be "removed" for free (the BOW/TFIDF matrix had lower rank than the number of words)




