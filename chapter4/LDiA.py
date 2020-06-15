# Latent Dirichlet Allocation uses two distributions for generating a bag representation of a document
# Poisson - estimate length of document
# Dirichlet - estimate number of topics in a document

# The Poisson parameter is simply estimated by the mean length of a document (in the bag of words representation)
# The K for the Dirichlet distribution is basically a free parameter to be tuned
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
import numpy as np
from nlpia.data.loaders import get_data
import pandas as pd

def get_sms_training_data():
    np.random.seed(42)
    sms = get_data('sms-spam')
    counter = CountVectorizer(tokenizer=casual_tokenize)
    index = ['sms{}{}'.format(i,'!'*j) for(i,j) in zip(range(len(sms)),sms.spam)]
    bow = pd.DataFrame(counter.fit_transform(raw_documents=sms.text).toarray(),index=index)
    cols, terms = zip(*sorted(zip(counter.vocabulary_.values(),counter.vocabulary_.keys())))
    # this one liner seems to say: sort counter.vocabulary_ by its values, then output those sorted values and keys.
    bow.columns = terms

    from sklearn.decomposition import LatentDirichletAllocation as LDiA
    mdl = LDiA(n_components=16,learning_method='batch')
    mdl = mdl.fit(bow)
    pd.set_option('display.width',75)
    col_names = ["topic"+str(i) for i in range(16)]
    comp = pd.DataFrame(mdl.components_.T,index=terms,columns=col_names)
    comp.round(2).head(3)
    comp.topic3.sort_values(ascending=False)[:10]
    topic_vecs = mdl.transform(bow)
    topic_vecs = pd.DataFrame(topic_vecs,index = index, columns = col_names)
    topic_vecs.round(2).head()
    return topic_vecs,sms.spam

def train_LDA():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import train_test_split
    vecs,labels = get_sms_training_data()
    X,X_test,y,y_test = train_test_split(vecs,labels,test_size = 0.5,random_state = 271828)
    lda = LDA(n_components=1)
    lda = lda.fit(X,y)
    acc = round(float(lda.score(X_test,y_test)),2)
    print(acc)
    return lda

def append_predictions():
    (vecs,_) = get_sms_training_data()
    mdl = train_LDA()
    sms = get_data('sms-spam') # importing this again...maybe the one giant script approach really is better...
    # ...
    # no, it's the children who are wrong
    sms['ldia_predict'] = mdl.predict(vecs)
    return sms

# i'm too lazy to continue following this - the book shows why tfidf would overfit here, hence motivating topic modelling
# tfidf overfits on an LDA classifier.


