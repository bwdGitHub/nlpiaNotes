import re
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer

def naiveTokenize(s):
    # naive tokenizer
    return str.split(s)

def regexTokenize(s):
    # in practice you may want to compile the regex, then call that compiled pattern, i.e.
    # >>> pattern = re.compile(someRegex)
    # >>> pattern.split(stuff)
    return re.split(r'[-\s.,;!?]+', s)

def nltkRegexTokenize(s):
    # wrapper for the nltk regex tokenizer using the book example
    t = RegexpTokenizer(r'\w+|$[0-9.]+|\S+')
    return t.tokenize(s)

def treebankTokenize(s):
    # wrapper for the nltk treebank tokenizer
    t = TreebankWordTokenizer()
    return t.tokenize(s)


# there is also nltk.tokenize.casual - intended for text data from something like twitter.