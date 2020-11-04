# Reference: https://github.com/liufly/delayed-memory-update-entnet

from __future__ import absolute_import

import json
import operator
import os
import re
import sys
import xml.etree.ElementTree

import nltk
import numpy as np


# original dataset (single location + double location)
def load_task(data_dir, aspect2idx):
    in_file = os.path.join(data_dir, 'sentihood-train.json')
    (train_labelled,train_unlabelled,aspect_count_tr) = parse_sentihood_json(in_file)
    in_file = os.path.join(data_dir, 'sentihood-dev.json')
    (dev_labelled,dev_unlabelled,aspect_count_dv) = parse_sentihood_json(in_file)
    in_file = os.path.join(data_dir, 'sentihood-test.json')
    (test_labelled,test_unlabelled,aspect_count_test) = parse_sentihood_json(in_file) 
    return (train_labelled,train_unlabelled,aspect_count_tr),(dev_labelled,dev_unlabelled,aspect_count_dv),(test_labelled,test_unlabelled,aspect_count_test)



#-----------------------------------------------------------------------------------------------------------------------------------------
# dataset divide to single and double location entities
def load_task_loc(data_dir):
    in_file = os.path.join(data_dir, 'sentihood-train.json')
    (tr_lab_single_loc,tr_lab_double_loc,tr_unlab_single_loc,tr_unlab_double_loc) = sep_loc1_loc2(in_file)
    in_file = os.path.join(data_dir, 'sentihood-dev.json')
    (dev_lab_single_loc,dev_lab_double_loc,dev_unlab_single_loc,dev_unlab_double_loc) = sep_loc1_loc2(in_file)
    in_file = os.path.join(data_dir, 'sentihood-test.json')
    (tst_lab_single_loc,tst_lab_double_loc,tst_unlab_single_loc,tst_unlab_double_loc) = sep_loc1_loc2(in_file)
    return (tr_lab_single_loc,tr_lab_double_loc,tr_unlab_single_loc,tr_unlab_double_loc),(dev_lab_single_loc,dev_lab_double_loc,dev_unlab_single_loc,dev_unlab_double_loc),(tst_lab_single_loc,tst_lab_double_loc,tst_unlab_single_loc,tst_unlab_double_loc)



#----------------------------------------------------------------------------------------------------------------------------------------
# test set preprocessing
def load_task_loc_test(data_dir):
    in_file = os.path.join(data_dir, 'sentihood-test.json')
    (lab_single_loc, lab_double_loc, single_labels, double_labels) = sep_loc1_loc2_test(in_file)
    return lab_single_loc, lab_double_loc, single_labels, double_labels


#-----------------------------------------------------------------------------------------------------------------------------------------
def sep_loc1_loc2_test(in_file):    
    with open(in_file) as f:
        data = json.load(f)
    lab_single_loc=[]
    lab_double_loc=[]
    single_labels=[]
    double_labels=[]
    
    def parse_sh_json(instance):
        aspect=[]
        opinions = []
        if len(d['opinions'])!=0:
            for opinion in d['opinions']:
                aspect.append(opinion['aspect'])                
            if instance=='single':
                lab_single_loc.append(text)
                single_labels.append(aspect)
            elif instance=='double':
                lab_double_loc.append(text)
                double_labels.append(aspect)
    for d in data:
        text = d['text']
        aspect=''
        target=['location1','location2']
        if 'location1' in text or 'location2' in text:
            parse_sh_json('single')
        #if all(x in text for x in target):
        if 'location1' in text and 'location2' in text:
            parse_sh_json('double')

    return lab_single_loc, lab_double_loc, single_labels, double_labels



#-----------------------------------------------------------------------------------------------------------------------------------------
def sep_loc1_loc2(in_file):
    
    with open(in_file) as f:
        data = json.load(f)
    lab_single_loc=[]
    lab_double_loc=[]
    unlab_single_loc=[]
    unlab_double_loc=[]
    
    def parse_sh_json(instance):
        aspect=[]
        opinions = []
        if len(d['opinions'])!=0:
            for opinion in d['opinions']:
                aspect.append(opinion['aspect'])
            if instance=='single':
                lab_single_loc.append(text)
            elif instance=='double':
                lab_double_loc.append(text)
        else:
            if instance=='single':
                unlab_single_loc.append(text)
            else:
                unlab_double_loc.append(text)
    for d in data:
        text = d['text']
        target=['location1','location2']
        if 'location1' in text or 'location2' in text:
            parse_sh_json('single')
        #if all(x in text for x in target):
        if 'location1' in text and 'location2' in text:
            parse_sh_json('double')

    return lab_single_loc,lab_double_loc,unlab_single_loc,unlab_double_loc



#_________________________________________________________________________________________________________________________________________
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



#-----------------------------------------------------------------------------------------------------------------------------------------
def get_aspect_idx(data, aspect2idx):
    ret = []
    for _, _, _, aspect, _ in data:
        ret.append(aspect2idx[aspect])
    assert len(data) == len(ret)
    return np.array(ret)



#-----------------------------------------------------------------------------------------------------------------------------------------
def parse_sentihood_json(in_file):
    with open(in_file) as f:
        data = json.load(f)

    review_text_labelled=[] #added to collect only the text -Sudeshna
    review_text_unlabelled=[]
    test_labels=[]
    
    ret = []
    unlabelled_ret=[]
    aspect_freq={}

    for d in data:
        text = d['text']
        sent_id = d['id']
        opinions = []
        targets = set()
        for opinion in d['opinions']:
            sentiment = opinion['sentiment']
            aspect = opinion['aspect']
            if aspect in aspect_freq.keys(): #to calculate aspect frequency
                aspect_freq[aspect]+=1
            else:
                aspect_freq[aspect]=1
            target_entity = opinion['target_entity']
            targets.add(target_entity)
            opinions.append((target_entity, aspect, sentiment))
        if len(d['opinions'])!=0:
            ret.append((sent_id, text, opinions))
            review_text_labelled.append(text)
        else:
            unlabelled_ret.append((sent_id, text, opinions))
            review_text_unlabelled.append(text)
    #return ret,unlabelled_ret
    print(review_text_labelled[:3])
    return review_text_labelled, review_text_unlabelled,aspect_freq #using only review text to train ABAE model - Sudeshna



#-----------------------------------------------------------------------------------------------------------------------------------------
def convert_input(data, all_aspects):
    ret = []
    for sent_id, text, opinions in data:
        for target_entity, aspect, sentiment in opinions:
            if aspect not in all_aspects:
                continue
            ret.append((sent_id, text, target_entity, aspect, sentiment))
        assert 'LOCATION1' in text
        targets = set(['LOCATION1'])
        if 'LOCATION2' in text:
            targets.add('LOCATION2')
        for target in targets:
            aspects = set([a for t, a, _ in opinions if t == target])
            none_aspects = [a for a in all_aspects if a not in aspects]
            for aspect in none_aspects:
                ret.append((sent_id, text, target, aspect, 'None'))
    return ret



#-----------------------------------------------------------------------------------------------------------------------------------------
def tokenize(data):
    ret = []
    for sent_id, text, target_entity, aspect, sentiment in data:
        new_text = nltk.word_tokenize(text)
        new_aspect = aspect.split('-')
        ret.append((sent_id, new_text, target_entity, new_aspect, sentiment))
    return ret
