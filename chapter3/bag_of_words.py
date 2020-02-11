from collections import Counter
class bag_of_words:
    # a very simple bag of words implementation

    def __init__(self,documents):
        # assume documents is a list of list of strs
        self.bag = []
        self.lexicon = set()
        self.lens = []

        for document in documents:
            self.bag.append(Counter(document))
            self.lens.append(len(document))
            for token in document:
                self.lexicon.add(token) 

    def tfPerDoc(self,word,i):
        return self.bag[i][word]/self.lens[i]       

    def tf(self,word):
        tf = []
        for i in range(len(self.bag)):
            tf.append(self.tfPerDoc(word,i))
        return tf