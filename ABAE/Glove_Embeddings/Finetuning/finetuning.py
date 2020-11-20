###############################################################################
# Train mittens model on cooccurrence and vocab from sentihood training corpus
# Zip vectors and new_embeddings to create finetuned embeddings
###############################################################################

import csv
import numpy as np
from mittens import Mittens
import pickle

def glove2dict(glove_filename):
    with open(glove_filename) as f:
        reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONE)
        embed = {line[0]: np.array(list(map(float, line[1:])))
                for line in reader}
    return embed


glove_pretrained="glove_wiki_100d.txt"
original_embeddings=glove2dict(glove_pretrained)

cooccurrence_matrix=pickle.load(open("cooccurrence/cooccurrence.pkl",'rb'))
vocab=pickle.load(open('cooccurrence/vocab.pkl','rb'))

# vocab=r"C:\Users\Sudeshna Dasgupta\Documents\Thesis_GuidedResearch\
#     thesis\Aspect Extraction\Unsupervised-Aspect-Extraction-master\
#     experiment_sentihood\code_py3_keras2compliant\GloveCode\
#     Glove_embeddings\trained on sentihood\GloVe-1.2\finetune_sent_vocab.txt"

mittens_model = Mittens(n=100, max_iter=1000)
new_embeddings = mittens_model.fit(
    cooccurrence_matrix,
    vocab=vocab,
    initial_embedding_dict= original_embeddings)
with open('new_embeddings.pkl','wb') as n:
    pickle.dump(new_embeddings,n)


newglove = dict(zip(vocab, new_embeddings))
f = open("repo_glove.pkl","wb")
pickle.dump(newglove, f)
f.close()