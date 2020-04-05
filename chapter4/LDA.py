# Linear Discrimnant Analysis (LDA) - a simple classification technique.
# Perform binary classification by finding the line between the centroids of the two classes
import pandas as pd
from nlpia.data.loaders import get_data

def get_sms_data():
    # the data is 4837 sms messages of which 638 are spam
    return get_data('sms-spam')

def preprocess():
    # get data into a dataframe
    sms = get_sms_data()
    index = ['sms{}{}'.format(i,'!'*j) for (i,j) in zip(range(len(sms)), sms.spam)]
    sms = pd.DataFrame(sms.values,columns = sms.columns,index=index)
    sms['spam'] = sms.spam.astype(int)
    return sms

    

