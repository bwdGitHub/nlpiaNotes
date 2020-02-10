# for whatever reason my conda env for the official nlpia repo is throwing MKL errors
# probably I set things up incorrectly.
# in any case all we need here is the movies dataset, so I copied it over to the same directory as this file.

# packages
import pandas as pd
from nltk.tokenize import casual_tokenize
from sklearn.naive_bayes import MultinomialNB
from collections import Counter

def getData(name):
    return pd.read_csv(name)

def bow(data):
    # create bow
    bow = []
    for text in data.text:
        bow.append(Counter(casual_tokenize(text)))
    return bow

def df(bow):
    df = pd.DataFrame.from_records(bow)
    return df.fillna(0).astype(int)

def train(df,y):

    nb = MultinomialNB()
    nb.fit(df,y)
    return nb

def mae(model,df,data):
    # this is specific to movies at the moment
    # the book seems to differ here - the model outputs a probability for each class
    # we only want the probability of "positive".
    # it's a little hard to retroactively work out which label that was
    pred = model.predict_proba(df)[:,1]*8 - 4
    err = (pred-data.sentiment).abs()
    return err.mean.round(1)

def workflow():
    data = getData('movieReviewSnippets_GroundTruth.csv.gz')
    bag = bow(data)
    frame = df(bag)
    # this isn't quite right and errors out - the usual thing, data in the wrong shape.
    nb = train(df,data.sentiment>0)
    e = mae(nb,frame,data)    
    print("The mean absolute error is {}".format(e))
    rightClass = (nb.predict(df)==(data.sentiment>0))
    print("The accuracy is {}".format(sum(rightClass)/len(data)))

