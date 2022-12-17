import twitter
import os
#from config import *
import time
import sys
import pickle
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import nltk
import pandas as pd


def pre_process(dataframe):
    processed=[]
    stop_words = set(stopwords.words('english') + list(punctuation) + ['AT_USER','URL'])

    for label,text in dataframe.values:
      text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text) # remove URLs
      text = re.sub('@[^\s]+', 'AT_USER', text) # remove usernames
      text = re.sub(r'#([^\s]+)', r'\1', text) # remove the # in #hashtag
      text = word_tokenize(text) # remove repeated characters (helloooooooo into hello)
      text=[w for w in text if w not in stop_words]
      processed.append([label,text])
    return processed

dataset=pd.read_csv("data/train.csv",encoding="latin-1",usecols=[0,5])
data=pd.DataFrame(dataset)
print(data.columns)
data.columns=['label','text']
print(data.columns)

lis=pre_process(data)
print(len(lis))
