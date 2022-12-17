import numpy as np
import pandas as pd
import pickle

embeddings_dict={}

with open("glove.twitter.27B.100d.txt","r",encoding="utf-8") as f:
    for line in f:
        values=line.split()
        word=values[0]
        vector=np.asarray(values[1:],"float32")
        embeddings_dict[word]=vector



k=embeddings_dict["king"]
print(k)
print(len(embeddings_dict))
pickle_out=open("embeddings_dict.pickle","wb")
pickle.dump(embeddings_dict,pickle_out)
pickle_out.close()
