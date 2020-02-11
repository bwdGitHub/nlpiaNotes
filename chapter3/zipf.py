from nltk.corpus import brown
from collections import Counter
import sys
import unicodedata
import matplotlib.pyplot as plt

def get_punctuation_characters():
    # get all punctuation characters
    # credit: https://stackoverflow.com/questions/11066400/remove-punctuation-from-unicode-formatted-strings
    return [chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')]

def remove_punctuation(strings):
    p = get_punctuation_characters()
    return [string for string in strings if string not in p]

# Zipf's law example - takes a few (~10-20) seconds
words = (x.lower() for x in remove_punctuation(brown.words()))
counts = Counter(words)
common = counts.most_common(50)
vals = [x[1] for x in common]
plt.plot(vals)
plt.show()


