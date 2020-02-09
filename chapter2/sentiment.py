from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def vaderLexicon():
    # wrapper for getting the lexicon from vaderSentiment
    sa = SentimentIntensityAnalyzer()
    return sa.lexicon

def sentiment(s):
    # wrapper for the vaderSentiment polarity scores.
    # note - this seems to work for strings without tokenization
    sa = SentimentIntensityAnalyzer()
    return sa.polarity_scores(text=s)