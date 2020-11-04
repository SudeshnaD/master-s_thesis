####Expansion of sentence with sememes
# get vocab index
# convert tensor to words
# expand to lemma
# update vocab
# get vocab index
# convert lemma to index
# extract embedding of whole vocab ----senth_emb_matrix
# weighted summation
# add to list of senses
# weighted summation
# add to list of words
# return word list
############################################################
import re
import os
from w2vEmbReader import W2VEmbReader as EmbReader
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
words = set(nltk.corpus.words.words())
import pickle

#load embedding model
# emb_name='../preprocessed_data/sentihood/glove.6B.100d.txt.word2vec'
# emb_reader = EmbReader(os.path.join("..", "preprocessed_data"), emb_name)


""" class Sememe:

    def __init__(self, vocab, batch_inp, emb_reader):
        self.vocab = vocab
        self.sen_idx_b = batch_inp
        self.emb_reader=emb_reader
        

    def weighted_sum(self, l_emb, E):
        x = np.array(l_emb)
        y = np.expand_dims(E, axis=-2)
        y = np.repeat(y, len(x), axis=-2)
        d=[]
        for i in range(x.shape[0]):
            d.append(np.tanh(np.dot(x[i],y[i])))
        sm=np.exp(d)/np.sum(np.exp(d))
        #sm = lambda i: np.exp(i)/np.sum(np.exp(d))
        # representation for all l sememes
        sm= np.reshape(sm,(sm.shape[0],1))
        x_ti=np.sum(sm*x, axis=0)
        return x_ti


    def weight_ext(self, vocab):
        senth_emb_matrix={}
        vocab_idx=[i for i in vocab.keys()]
        for word in vocab_idx: #vocab_idx=list of vocab words or vocab dict keys
            try:
                senth_emb_matrix[word]=self.emb_reader.embeddings[word]
            except KeyError:
                senth_emb_matrix[word]=[0]*100
        senth_emb_v=list(senth_emb_matrix.values())
        return senth_emb_v


    def Sen_Average(self, sen_idx):
        senth_emb_v=self.weight_ext(self.vocab)
        sen_emb=[]
        for word in sen_idx: #list of tokens
            sen_emb.append(senth_emb_v[word])
        sen_emb=np.array(sen_emb)
        E_avg=np.mean(sen_emb,axis=0)
        S=self.weighted_sum(sen_emb,E_avg)
        return S


    def emb_ext(self, l_idx, senth_emb_v):
        lemma_vecs = []
        for word in l_idx: #list of tokens
            lemma_vecs.append(senth_emb_v[word])
        return lemma_vecs


    def vocab_up(self, lemma, vocab):
        for w in lemma:
            if w not in vocab.keys():
                vocab[w]=1
        return vocab


    def sememe_expansion(self, sentence_input, S):
        batch_input=[]
        sentence=[]
        vocab=self.vocab
        vocab_idx=[i for i in vocab.keys()]
        #for sen in sentence_input:
        for i in sentence_input:
            sentence.append(str(vocab_idx[i]))
        et_sentence=[]
        x_t=[]
        for i in sentence: 
            if i in words:
                for j in wn.synsets(i):
                    l=wn.synset(j.name()).lemma_names()
                    vocab=self.vocab_up(l,vocab)
                    vocab_idx=[i for i in vocab.keys()]
                    l_idx=[vocab_idx.index(i) for i in l]
                    emb_w=self.weight_ext(vocab)
                    l_emb=self.emb_ext(l_idx,emb_w)
                    x_ti=self.weighted_sum(l_emb,S)
                    x_t.append(x_ti)  # all synsets
            e_t=self.weighted_sum(x_t,S) ##########output from equation 8
            et_sentence.append(e_t)
        #    batch_input.append(e_t_sentence)
        #return batch_input,vocab
        return et_sentence, vocab


    def run_script(self):
        sen_lemma_b=[]
        batch_average=[]
        vocab=self.vocab
        for sen_idx in self.sen_idx_b:
            S = self.Sen_Average(sen_idx)
            sen_lemma, vocab = self.sememe_expansion(sen_idx, S)
            sen_lemma_b.append(sen_lemma)
            batch_average.append(S)
        return sen_lemma_b, batch_average, vocab """


#----------------------------------------------------------------------------------------------------------------

class Sememe:

    def __init__(self, vocab, batch_inp, emb_reader):
        vocab_d={}
        with open('vocab_c.pkl','rb') as f:
            vocab_c=pickle.load(f)
        self.vocab = vocab_c
        self.sen_idx_b = batch_inp
        self.emb_reader=emb_reader
        

    def weighted_sum(self, l_emb, E):
        x = np.array(l_emb)
        y = np.expand_dims(E, axis=-2)
        y = np.repeat(y, len(x), axis=-2)
        d=[]
        for i in range(x.shape[0]):
            d.append(np.tanh(np.dot(x[i],y[i])))
        sm=np.exp(d)/np.sum(np.exp(d))
        #sm = lambda i: np.exp(i)/np.sum(np.exp(d))
        # representation for all l sememes
        sm= np.reshape(sm,(sm.shape[0],1))
        x_ti=np.sum(sm*x, axis=0)
        return x_ti


    def weight_ext(self, vocab):
        senth_emb_matrix={}
        vocab_idx=[i for i in vocab.keys()]
        for word in vocab_idx: #vocab_idx=list of vocab words or vocab dict keys
            try:
                senth_emb_matrix[word]=self.emb_reader.embeddings[word]
            except KeyError:
                senth_emb_matrix[word]=[0]*100
        senth_emb_v=list(senth_emb_matrix.values())
        return senth_emb_v


    def Sen_Average(self, sen_idx):
        senth_emb_v=self.weight_ext(self.vocab)
        sen_emb=[]
        for word in sen_idx: #list of tokens
            sen_emb.append(senth_emb_v[word])
        sen_emb=np.array(sen_emb)
        E_avg=np.mean(sen_emb,axis=0)
        S=self.weighted_sum(sen_emb,E_avg)
        return S


    def emb_ext(self, l_idx, senth_emb_v):
        lemma_vecs = []
        for word in l_idx: #list of tokens
            lemma_vecs.append(senth_emb_v[word])
        return lemma_vecs


    def vocab_up(self, lemma, vocab):
        for w in lemma:
            if w not in vocab.keys():
                vocab[w]=1
        return vocab


    def sememe_expansion(self, sentence_input, S):
        batch_input=[]
        sentence=[]
        vocab=self.vocab
        vocab_idx=[i for i in vocab.keys()]
        #for sen in sentence_input:
        for i in sentence_input:
            sentence.append(str(vocab_idx[i]))
        et_sentence=[]
        x_t=[]
        for i in sentence: 
            if i not in words:
                e_t=np.zeros(100)

            elif len(wn.synsets(i))==0:
                e_t=np.zeros(100)
            else:
                for j in wn.synsets(i):
                    l=wn.synset(j.name()).lemma_names()
                    vocab=self.vocab_up(l,vocab)
                    vocab_idx=[i for i in vocab.keys()]
                    l_idx=[vocab_idx.index(i) for i in l]
                    emb_w=self.weight_ext(vocab)
                    l_emb=self.emb_ext(l_idx,emb_w)
                    x_ti=self.weighted_sum(l_emb,S)
                    x_t.append(x_ti)
                e_t=self.weighted_sum(x_t,S) ##########output from equation 8
            if np.array(e_t).shape==(0,):
                print(i)
            et_sentence.append(e_t)
        #    batch_input.append(e_t_sentence)
        #return batch_input,vocab
        return et_sentence


    def run_script(self):
        sen_lemma_b=[]
        batch_average=[]
        for sen_idx in self.sen_idx_b:
            S = self.Sen_Average(sen_idx)
            sen_lemma = self.sememe_expansion(sen_idx, S)
            sen_lemma_b.append(sen_lemma)
            batch_average.append(S)
        return sen_lemma_b, batch_average







