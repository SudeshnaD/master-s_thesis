
import argparse
from w2vEmbReader import W2VEmbReader as EmbReader
from keras.layers import Input, Embedding
from keras.models import Model
from keras.constraints import MaxNorm
import numpy as np
import keras.backend as K
import os


parser=argparse.ArgumentParser()
parser.add_argument("-as", "--aspect_size", dest="aspect_size", type=int, metavar='<int>', default=14,
                    help="The number of aspects specified by users (default=14)")
parser.add_argument("-asrange", "--aspect_range", nargs='+', dest="aspect_range", type=int, metavar='<int>', default=14,help="Range of aspects specified by users (default=14)")
parser.add_argument("--emb-name",  type=str, help="The name to the word embeddings file", default="w2v_64k_unigram_100d.model")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>',
                        help="The path to the output directory", default="output")
args=parser.parse_args()

out_dir = args.out_dir_path

import pickle
with open('vocab_c.pkl','rb') as f:
    vocab=pickle.load(f)


emb_reader = EmbReader(os.path.join("..", "preprocessed_data"), args.emb_name)

for k in range(args.aspect_range[0],args.aspect_range[1]+1,10):
    args.aspect_size=k
    aspect_matrix = emb_reader.get_aspect_matrix(args.aspect_size)


    sym = Input(shape=(None,), dtype='int32', name='sym')
    word_emb_l = Embedding(len(vocab), 100, mask_zero=True, name='word_emb_l', embeddings_constraint=MaxNorm(10))
    e_sym = word_emb_l(sym)
    mod = Model(inputs=[sym], outputs=[e_sym])
    embs = mod.get_layer('word_emb_l').embeddings
    K.set_value(embs, emb_reader.get_emb_matrix_given_vocab(vocab, K.get_value(embs)))
    word_emb = K.get_value(mod.get_layer('word_emb_l').embeddings)
    word_emb = word_emb / np.linalg.norm(word_emb, axis=-1, keepdims=True)

    aspect_file = open(out_dir + '/aspect.log{}'.format(args.aspect_size), 'wt', encoding='utf-8')

    aspect_emb=aspect_matrix 

    vocab_inv = {}
    for w, ind in vocab.items():
        vocab_inv[ind] = w

    for ind in range(len(aspect_emb)):
        desc = aspect_emb[ind]
        sims = word_emb.dot(desc.T) #cosine similarity
        ordered_words = np.argsort(sims)[::-1] #sort max to min
        desc_list = [vocab_inv[w] + "|" + str(sims[w]) for w in ordered_words[:100]]
        aspect_file.write('Aspect %d:\n' % ind)
        aspect_file.write(' '.join(desc_list) + '\n\n')