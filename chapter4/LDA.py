# Linear Discrimnant Analysis (LDA) - a simple classification technique.
# Perform binary classification by finding the line between the centroids of the two classes
import pandas as pd
from nlpia.data.loaders import get_data
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

def get_sms_data():
    # the data is 4837 sms messages of which 638 are spam
    return get_data('sms-spam')

def preprocess():
    # get data into a dataframe
    sms = get_sms_data()
    index = ['sms{}{}'.format(i,'!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]
    sms = pd.DataFrame(sms.values,columns = sms.columns,index=index)
    sms['spam'] = sms.spam.astype(int)
    return sms

def encode(data):
    mdl = TfidfVectorizer(tokenizer = casual_tokenize)
    return mdl.fit_transform(raw_documents=data).toarray()

# Note - the vocab of 'sms-spam' is 9232 words, with 638 spam messages.
# # A model like Naive Bayes may not do well here because the vocab is to high whilst number of examples with the spam label is propotionally low.

def get_centroid(data):
    # return the centroid of data
    return data.mean(axis=0)

def get_masked_centroids(data,mask):
    # return two centroids, one where mask is true and one where mask is false
    return [get_centroid(data[mask]),get_centroid(data[~mask])]

def get_LDA_line(data,mask):
    # return difference of two centroids - corresponds to a line
    # note: should probably be normalized in practice
    centroids = get_masked_centroids(data,mask)
    return centroids[0] - centroids[1]

def spam_score(encoded,line):
    # score by projecting onto the line between the centroids.
    # essentially - measure closeness to centroid ignoring orthogonal directions to the centroid difference
    return encoded.dot(line)

def sms_spam_score(encoded):
    # apply spam_score to encoded using sms data
    sms = preprocess()
    data = encode(sms.text)
    mask = sms.spam.astype(bool).values
    line = get_LDA_line(data,mask)
    return spam_score(encoded,line)

def sms_self_spam_score():
    # apply sms_spam_score to the dataset itself.
    sms = preprocess()
    data = encode(sms.text)
    return sms_spam_score(data)