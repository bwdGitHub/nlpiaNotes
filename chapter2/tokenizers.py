import re

def naiveTokenize(s):
    # naive tokenizer
    return str.split(s)

def regexTokenize(s):
    return re.split(r'[-\s.,;!?]+', s)