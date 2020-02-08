# naive bag of words
from collections import Counter
import numpy as np
str = input()
bag = Counter(str.split())
for item in bag:
    print("Token:{} \nwith frequency:{}".format(item,bag[item]))

# bagofwords vectors
vec = np.zeros(len(bag))
for (i,item) in zip(range(len(bag)),bag.keys()):
    vec[i] = bag[item]

print("The vector represention is:\n {}".format(vec))