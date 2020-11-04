finetuning.py: Script for finetuning GloVe embeddings.
finetune_sent_vocab: Vocabulary file.
new_embeddings_weightsonly.pkl: Embedding weights obtained by finetuning using Mittens.
repo_glove.pkl: Finetuned embeddings in dictionary data type.
cooccurrence/cooccurrence.py: Script to create cooccurrence matrix.
--------------------------------------------------------------------------------------------------------------------------------------
##how to convert the repoglove.pkl to word to vec format?

-----1)convert embedding dictionary to the glove embedding txt file format:
flat_a=[x+' '+' '.join(str(i) for i in y) for x,y in a.items()]

-----2) convert text to word2vec using glove2word2vec.py
