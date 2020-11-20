Keep glove pretrained embeddings file here. 
File should be converted to word2vec format using glove2word2vec.py.
(Not included in repository due to file size.)
-----------------------------------------------------------------------------------------------------------------------------------------------------------
train.txt==train_dev_labelled_unlabelled.txt
test.txt==labelled data from sentihood_test.json

test_filtered: test set without sentences with low precision aspect labels
test_labels_filtered: labels for test_filtered.

vocab: vocab saved in pickle format.

w2v_embedding: word2vec embeddings trained on Sentihood dataset.