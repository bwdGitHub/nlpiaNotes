import pandas as pd
pd.set_option('display.max_columns',6)
from sklearn.decomposition import PCA
import seaborn
from matplotlib import pyplot as plt
from nlpia.data.loaders import get_data

def horse_plot():
    df = get_data('pointcloud').sample(1000)
    pca = PCA(n_components=2)
    df2d = pd.DataFrame(pca.fit_transform(df),columns=list('xy'))
    df2d.plot(kind='scatter',x='x',y='y')
    plt.show()

def get_sms_data():
    pd.options.display.width=120
    sms = get_data('sms-spam')
    index = ['sms{}{}'.format(i,'!'*j) for(i,j) in zip(range(len(sms)),sms.spam)]
    sms.index = index
    sms.head(6)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.tokenize.casual import casual_tokenize
    tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
    tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
    tfidf_docs = pd.DataFrame(tfidf_docs)
    tfidf_docs = tfidf_docs - tfidf_docs.mean() # mean centering
    return (tfidf_docs,sms,tfidf)

def sms_pca(tfidf,sms,n_components = 16):
    pca = PCA(n_components=n_components)
    pca = pca.fit(tfidf)
    topics = pca.transform(tfidf)
    cols = ['topic{}'.format(i) for i in range(pca.n_components)]
    index = ['sms{}{}'.format(i,'!'*j) for(i,j) in zip(range(len(sms)),sms.spam)]
    topics = pd.DataFrame(topics,columns = cols, index = index)
    return (topics,pca)

def get_topic_weights()
    tfidf,sms,model = get_sms_data()
    topics,pca = sms_pca(tfidf,sms)
    col_nums,terms = zip(*sorted(zip(model.vocabulary_.values(),model.vocabulary_.keys())))
    w = pd.DataFrame(pca.components_,columns = terms, index = ['topic{}'.format(i) for i in range(16)])
    return w

