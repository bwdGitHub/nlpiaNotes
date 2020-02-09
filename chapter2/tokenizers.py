import re

def naiveTokenize(s):
    # naive tokenizer
    return str.split(s)

def regexTokenize(s):
    # in practice you may want to compile the regex, then call that compiled pattern, i.e.
    # >>> pattern = re.compile(someRegex)
    # >>> pattern.split(stuff)
    return re.split(r'[-\s.,;!?]+', s)