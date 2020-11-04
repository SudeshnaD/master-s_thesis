import os
import numpy as np
import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from keras.constraints import MaxNorm

import gensim
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)



class Embedding_Ext:

    def __init__(self, vocab, sen_input, emb_reader):
        #emb_name = '..\preprocessed_data\sentihood\glove.6B.100d.txt.word2vec'
        #data_path = os.path.join("..", "preprocessed_data")
        #if os.path.sep not in emb_name:
        #    emb_path = os.path.join(data_path, emb_name)
        #else:
        #    emb_path = emb_name
        self.embeddings = emb_reader.embeddings
        #model=gensim.models.KeyedVectors.load_word2vec_format(emb_path, binary=False)

        #emb_dim = model.vector_size
        #for word in model.wv.vocab:
        #    self.embeddings[word] = list(model[word])
        self.vocab=vocab
        self.sen_inp=sen_input
        
        


    def get_emb_matrix_vocab(self, emb_matrix):
            counter = 0
            for word, index in self.vocab.items():
                try:
                    emb_matrix[int(index)] = self.embeddings[word]
                    counter += 1
                except KeyError:
                    pass
            logger.info(
            '%i/%i word vectors initialized (hit rate: %.2f%%)' % (counter, len(self.vocab), 100 * counter / len(self.vocab)))
            # L2 normalization
            norm_emb_matrix = emb_matrix / np.linalg.norm(emb_matrix, axis=-1, keepdims=True)
            return norm_emb_matrix


    

    def run_script(self):
            sen_sym = Input(shape=(69, ), dtype='int32', name='sen_sym')
            word_emb = Embedding(len(self.vocab), 100, mask_zero=True, name='word_emb', embeddings_constraint=MaxNorm(10))
            e_sen = word_emb(sen_sym)
            model = Model(inputs=[sen_sym], outputs=[e_sen])
            embs = model.get_layer('word_emb').embeddings
            K.set_value(embs, self.get_emb_matrix_vocab(K.get_value(embs)))
            sen_input=K.cast(self.sen_inp, 'float32')
            sen_emb=K.eval(word_emb(sen_input))
            return sen_emb
            # ng_means=[]
            # for j in range(ng_emb.shape[0]):
            #     ng_means.append([np.mean(ng_emb[j][i], axis=0) for i in range(ng_emb.shape[1])])
            # return ng_means

