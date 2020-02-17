# script to try out already implemented library versions of the concepts so far (i.e. bag of words and tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
# one accepted syntax is a list of strings.
# not clear what the tokenizer is but can be passed in.
corpus = [
    "foo bar and baz",
    "foo again but not the rest",
    "baz is back, legend",
]
vectorizer = TfidfVectorizer(min_df=1)
mdl = vectorizer.fit_transform(corpus)
# some other idf options are supported, but seemingly not the full list of Table 3.1 in the book.
print(mdl.todense().round(2))
