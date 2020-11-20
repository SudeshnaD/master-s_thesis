from gensim.scripts.glove2word2vec import glove2word2vec


glove_input_file = '..\Glove_embeddings\pretrained\glove.6B\glove.6B.200d.txt'
word2vec_output_file = '..\preprocessed_data\sentihood\glove6B200d.word2vec'

#The first step is to convert the GloVe file format to the word2vec file format.
glove2word2vec(glove_input_file, word2vec_output_file)