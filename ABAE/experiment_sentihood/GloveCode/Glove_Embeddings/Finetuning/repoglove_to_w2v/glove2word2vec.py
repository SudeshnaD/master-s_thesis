from gensim.scripts.glove2word2vec import glove2word2vec


glove_input_file = 'flat_repoglove.txt'
word2vec_output_file = 'flat_repoglove.txt.word2vec'

#The first step is to convert the GloVe file format to the word2vec file format.
glove2word2vec(glove_input_file, word2vec_output_file)