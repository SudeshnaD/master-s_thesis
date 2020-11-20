

# Reference: https://github.com/liufly/delayed-memory-update-entnet
##############################################################
# Creates a cleaned test dataset for required labels only
##############################################################


from __future__ import absolute_import

import json
import operator
import os
import re
import sys
import xml.etree.ElementTree
import nltk
import numpy as np
#import argparse
import pickle
from sentihood_preprocess import parseSentence



data_dir='../data/sentihood/'
aspect2idx = {
    'general': 0,
    'price': 1,
    'transit-location': 2,
    'safety': 3,
}

# parser=argparse.ArgumentParser()
# parser.add_argument('-af','--aspect_filter',type='')

def load_task(data_dir, aspect2idx):
    print('in load task')
    in_file = os.path.join(data_dir, 'sentihood-test.json')
    filter_aspect=['dining','green-nature','green-culture','live','quiet','touristy']
    parse_sentihood_aspect_based(in_file,filter_aspect)
    #extract_label(in_file)
    return None


def extract_label(in_file):
    with open(in_file) as f:
        data = json.load(f)
    test_set=[]
    test_labels=[]
    for d in data:
        text = d['text']
        opinions = []
        aspect=''
        for opinion in d['opinions']:
            aspect = opinion['aspect']
        if len(d['opinions'])!=0:
            test_set.append((text,aspect))
            test_labels.append(aspect)
    print('No. of data in test set: ',len(test_set),len(test_labels))
    out=open("test data_labels/test_set.txt", "w", encoding='utf-8')
    out.write(str(test_set))
    with open("test data_labels/test_labels.txt", "w", encoding='utf-8') as out:
        out.write("\n".join(str(i) for i in test_labels))
    return None


def parse_sentihood_aspect_based(in_file,filter_aspect):
    print('in parse_sentihood_aspect_based')
    with open(in_file) as f:
        data = json.load(f)
    test_set_filtered=[]
    test_labels_filtered=[]
    for d in data:
        text = d['text']
        opinions = []
        aspect=''
        for opinion in d['opinions']:
            aspect = opinion['aspect']
        if len(d['opinions'])!=0:
            if aspect not in filter_aspect :
                test_set_filtered.append(text)
                test_labels_filtered.append(aspect)
    print('No. of data in filtered test set: ',len(test_set_filtered),len(test_labels_filtered))
    with open('test_set_filtered', 'ab') as pfile:
        pickle.dump(test_set_filtered,pfile)
    with open("test_set_filtered.txt", "w", encoding='utf-8') as out:
        out.write(str(test_set_filtered))
    with open("test_labels_filtered.txt", "w", encoding='utf-8') as out:
        out.write("\n".join(str(i) for i in test_labels_filtered))  
    return None
    #return ret,unlabelled_ret
    #print(review_text_labelled[:3])
    #return review_text_labelled, review_text_unlabelled,aspect_freq #using only review text to train ABAE model - Sudeshna


if __name__=='__main__':
    load_task(data_dir,aspect2idx)

    with open("test_filtered_ppsd.txt", "w", encoding='utf-8') as out:
        with open('test_set_filtered', 'rb') as infile:  
            ts_filtered = pickle.load(infile)
        for line in ts_filtered:
            tokens = parseSentence(line)
            if len(tokens) > 0:
                out.write(' '.join(tokens)+'\n')
"""         with open("test_set_filtered.txt", "r", encoding='utf-8') as infile:
            ts_filtered=infile.readlines()
            print(ts_filtered) """


