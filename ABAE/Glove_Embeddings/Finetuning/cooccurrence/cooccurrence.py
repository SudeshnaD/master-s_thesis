######################################################
# Calculate cooccurrence and vocab for a window size
######################################################

from collections import defaultdict
import pandas as pd
import numpy as np
import argparse
import nltk
from nltk.corpus import wordnet 
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--window", dest="window", type=int, metavar='<int>', default=10,
                    help="cooccurrence window")
args = parser.parse_args()

def co_occurrence(sentences, window_size):
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        # preprocessing (use tokenizer instead)
        text = text.lower().split()
        #remove non-alphabetical tokens
        #cleaned_text=[w for w in text if w.isalpha()==True] 
        #cleaned_text=[w for w in text if w.isalpha()==True & len(wordnet.synsets(w))!=0]
        cleaned_text=[w for w in text if w.isalpha()==True]
        # iterate over sentences
        for i in range(len(cleaned_text)):
            token = cleaned_text[i]
            vocab.add(token)  # add to vocab
            next_token = cleaned_text[i+1 : i+1+window_size]
            for t in next_token:
                key = tuple( sorted([t, token]) )
                d[key] += 1

    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    
    print(len(vocab))
    #print to view the matrix
    print(df.head())

    #save to binary format
    cooccurrence = df.to_numpy()
    return cooccurrence,vocab
    
    

# docs=['along location1 lot electronics shop independent one',
# 'location1 ten min direct tube location2',
# 'another option location1 central ton club bar within walking distance']

if __name__=='__main__':

 with open('train.txt','r') as s:
     doc=s.readlines()
     doc=[t.strip('\n') for t in doc]

 co_occurrence,vocab=co_occurrence(doc,args.window)

 with open("vocab.pkl","wb") as f:
    pickle.dump(vocab, f)
 
 with open("cooccurrence.pkl","wb") as f:
    pickle.dump(co_occurrence, f)
 
 #np.save('vocab',vocab)
 #np.save('cooccurrence',co_occurrence)
