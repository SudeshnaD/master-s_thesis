import argparse
from w2vEmbReader import W2VEmbReader as EmbReader
from keras.layers import Input, Embedding
from keras.models import Model
from keras.constraints import MaxNorm
import numpy as np
import keras.backend as K
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import classification_report,precision_score,confusion_matrix
import keras.backend as K
from keras.preprocessing import sequence

import utils as U
import reader as dataset

#parser=argparse.ArgumentParser()
parser = U.add_common_args()
parser.add_argument("-as", "--aspect_size", dest="aspect_size", type=int, metavar='<int>', default=14,
                    help="The number of aspects specified by users (default=14)")
parser.add_argument("-asrange", "--aspect_range", nargs='+', dest="aspect_range", type=int, metavar='<int>', default=14,help="Range of aspects specified by users (default=14)")
parser.add_argument("--emb-name",  type=str, help="The name to the word embeddings file", default="w2v_64k_unigram_100d.model")
#parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>',
#                        help="The path to the output directory", default="output")
args=parser.parse_args()

#out_dir = args.out_dir_path

vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)

import pickle
with open('vocab_c.pkl','rb') as f:
    vocab=pickle.load(f)


emb_reader = EmbReader(os.path.join("..", "preprocessed_data"), args.emb_name)



sym = Input(shape=(None,), dtype='int32', name='sym')
word_emb_l = Embedding(len(vocab), 100, mask_zero=True, name='word_emb_l', embeddings_constraint=MaxNorm(10))
e_sym = word_emb_l(sym)
mod = Model(inputs=[sym], outputs=[e_sym])
embs = mod.get_layer('word_emb_l').embeddings
K.set_value(embs, emb_reader.get_emb_matrix_given_vocab(vocab, K.get_value(embs)))
word_emb = K.get_value(mod.get_layer('word_emb_l').embeddings)
word_emb = word_emb / np.linalg.norm(word_emb, axis=-1, keepdims=True)

sent_avg_emb=[]
for sentence in test_x:
    print(sentence)
    sent_emb=[]
    for word in sentence:
        sent_emb.append(emb_reader.embeddings[word])
    print(sent_emb)
    sent_avg_emb.append(sum(sent_emb)/len(sent_emb))


with open('sent_emb.txt','w') as f:
    f.write(str(sent_avg_emb))

with open('sent_emb.pkl','wb') as f:
    pickle.dump(f,sent_avg_emb)
