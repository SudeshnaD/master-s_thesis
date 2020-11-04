#from os import *
import codecs
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


def parseSentence(line):
    lmtzr = WordNetLemmatizer()    
    stop = stopwords.words('english')
    text_token = CountVectorizer().build_tokenizer()(line.lower())
    text_rmstop = [i for i in text_token if i not in stop]
    text_stem = [lmtzr.lemmatize(w) for w in text_rmstop]
    return text_stem


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