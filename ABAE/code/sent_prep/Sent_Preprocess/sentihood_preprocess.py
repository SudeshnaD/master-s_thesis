#from os import *
import os
import codecs
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer




#----------------------------------------------------------------------------------------------------------------------------------------
words = set(nltk.corpus.words.words())

def parseSentence(line):
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    #cleaned_text=[w for w in text_stem if w.isalpha()==True if len(w)>3 if w.lower() in words] #added to remove alphanumericals and words smaller than 3 letters.
    #return cleaned_text
    return text_stem


#--------------------------------------------------------------------------------------------------------------------------------------
def dataset_creation(instance,sh_flt_train,sh_flt_dev,sh_flt_test):
    # to train model on train+dev data
    train_labelled_unlabelled= sh_flt_train[0]+sh_flt_train[1]
    dev_labelled_unlabelled= sh_flt_dev[0]+sh_flt_dev[1]
    train_dev_labelled_unlabelled= train_labelled_unlabelled+dev_labelled_unlabelled
    if not os.path.exists(instance+'/'):
        os.mkdir(instance+'/')
    path=instance+'/'+"train_dev_labelled_unlabelled.txt"
    #with open(instance+'/'+"train_dev_labelled_unlabelled.txt", "w", encoding='utf-8') as out:
    with open(path, "w", encoding='utf-8') as out:
        for line in train_dev_labelled_unlabelled:
            tokens = parseSentence(line)
            if len(tokens) > 0:
                out.write(' '.join(tokens)+'\n')
    print('training data for {} location: {}, dev data: {}, total: {}'.format(instance, len(train_labelled_unlabelled), len(dev_labelled_unlabelled), len(train_dev_labelled_unlabelled)))

    #saving labelled test data for evaluation
    labelled_text_test=sh_flt_test[0]
    test_labels=sh_flt_test[1]
    path=instance+'/'+"labelledTest.txt"
    with open(path, "w", encoding='utf-8') as out:
        for x in labelled_text_test:
            out.write(str(x))
    path=instance+'/'+"labelledTest_labels.txt"
    with open(path, "w", encoding='utf-8') as out:
        for x in test_labels:
            out.write(str(x))
    """ out=open(instance+'/'+"labelledTest.txt", "w", encoding='utf-8') # test dataset does not need cleaning
    for line in labelled_text_test:
        tokens = parseSentence(line)
        if len(tokens) > 0:
            out.write(' '.join(tokens)+'\n') """


#---------------------------------------------------------------------------------------------------------------------------------------
# for single location
pfile = open('pickled/single/sh_flt_train', 'rb')      
sh_flt_train = pickle.load(pfile) 
pfile.close()
pfile = open('pickled/single/sh_flt_dev', 'rb')      
sh_flt_dev = pickle.load(pfile) 
pfile.close()
pfile = open('pickled/single/sh_flt_testnlabel', 'rb')      
sh_flt_test = pickle.load(pfile)
pfile.close()
# pfile = open('pickled/single/sh_flt_test', 'rb')      
# sh_flt_test = pickle.load(pfile) 
# pfile.close()
dataset_creation('single',sh_flt_train,sh_flt_dev,sh_flt_test)


#---------------------------------------------------------------------------------------------------------------------------------------
# for double location
pfile = open('pickled/double/sh_flt_train', 'rb')      
sh_flt_train = pickle.load(pfile) 
pfile.close()
pfile = open('pickled/double/sh_flt_dev', 'rb')      
sh_flt_dev = pickle.load(pfile) 
pfile.close()
pfile = open('pickled/double/sh_flt_test', 'rb')      
sh_flt_test = pickle.load(pfile) 
pfile.close()
dataset_creation('double',sh_flt_train,sh_flt_dev,sh_flt_test)


#--------------------------------------------------------------------------------------------------------------------------------------
# original mixed dataset

pfile = open('pickled/sh_flt_train', 'rb')      
sh_flt_train = pickle.load(pfile) 
pfile.close()
pfile = open('pickled/sh_flt_dev', 'rb')      
sh_flt_dev = pickle.load(pfile) 
pfile.close()
pfile = open('pickled/sh_flt_test', 'rb')      
sh_flt_test = pickle.load(pfile) 
pfile.close()

# to train model on train+dev data
train_labelled_unlabelled= sh_flt_train[0]+sh_flt_train[1]
dev_labelled_unlabelled= sh_flt_dev[0]+sh_flt_dev[1]
train_dev_labelled_unlabelled= train_labelled_unlabelled+dev_labelled_unlabelled
out=open("train_dev_labelled_unlabelled.txt", "w", encoding='utf-8')
for line in train_dev_labelled_unlabelled:
    tokens = parseSentence(line)
    if len(tokens) > 0:
        out.write(' '.join(tokens)+'\n')

print('training data: %d, dev data: %d, total: %d' % (len(train_labelled_unlabelled),len(dev_labelled_unlabelled),len(train_dev_labelled_unlabelled)))


#saving labelled test data for evaluation
labelled_text_out_test=sh_flt_test[0]
out=open("labelledTest.txt", "w", encoding='utf-8')
for line in labelled_text_out_test:
    tokens = parseSentence(line)
    if len(tokens) > 0:
        out.write(' '.join(tokens)+'\n')


# saving unlabelled data 
unlabelled_text_out_train=sh_flt_train[1]
unlabelled_text_out_dev=sh_flt_dev[1]
unlabelled_text_out_test=sh_flt_test[1]
out=open("unlabelledTrain.txt", "w", encoding='utf-8')
for line in unlabelled_text_out_train:
    tokens = parseSentence(line)
    if len(tokens) > 0:
        out.write(' '.join(tokens)+'\n')

out=open("unlabelledDev.txt", "w", encoding='utf-8')
for line in unlabelled_text_out_dev:
    tokens = parseSentence(line)
    if len(tokens) > 0:
        out.write(' '.join(tokens)+'\n')

out=open("unlabelledTest.txt", "w", encoding='utf-8')
for line in unlabelled_text_out_test:
    tokens = parseSentence(line)
    if len(tokens) > 0:
        out.write(' '.join(tokens)+'\n')


""" with open("unlabelled_text_out.txt", "r", 'UTF-8') as f:
    with open("Test_unlabelled.txt", "w", 'utf-8') as f_out:
        for line in f:
            tokens = parseSentence(line)
            if len(tokens) > 0:
                f_out.write(' '.join(tokens)+'\n') """


#f.write(str(train_labelled))