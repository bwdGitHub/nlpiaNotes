def naiveCaseNormalization(document):
    # simple normalization
    return [token.lower() for token in document]

def slightlyLessNaiveNormalization(document):
    # only normalize the first tokens in a sentence
    # really this needs a sentence-end detector.
    # also proper nouns should be detected up front and preserved.
    for i in range(len(document)-1):
        if i==0 or document[i-1]==".":
            document[i] = document[i].lower()
    return document


