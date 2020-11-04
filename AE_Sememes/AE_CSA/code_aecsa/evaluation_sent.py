import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import classification_report,precision_score,confusion_matrix
import keras.backend as K
from keras.preprocessing import sequence

import utils as U
import reader as dataset
from my_layers import SememeAttention, Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin

######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training

parser = U.add_common_args()
parser.add_argument("--embname",  type=str,
                    help="The name to the word embeddings file", default="w2v_64k_unigram_100d.model")
args = parser.parse_args()

#out_dir = args.out_dir_path + '/' + args.domain
#out_dir = '../pre_trained_model/' + args.domain
out_dir = args.out_dir_path #changed_SD
U.print_args(args)

# assert args.domain in {'restaurant', 'beer'}

###### Get test data #############
#vocab, train_x, _, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)
#changed_SD
vocab, train_x, test_x, overall_maxlen = dataset.get_data(args.domain, vocab_size=args.vocab_size, maxlen=args.maxlen)


test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)
test_length = test_x.shape[0]


splits = []
for i in range(1, test_length // args.batch_size):
    splits.append(args.batch_size * i)
if test_length % args.batch_size:
    splits += [(test_length // args.batch_size) * args.batch_size]
test_x = np.split(test_x, splits)



############# Build model architecture, same as the model used for training #########

## Load the save model parameters
model = load_model(out_dir + '/model_param',
                   custom_objects={"SememeAttention": SememeAttention, "WeightedSum": WeightedSum,
                                   "MaxMargin": MaxMargin, "WeightedAspectEmb": WeightedAspectEmb,
                                   "max_margin_loss": U.max_margin_loss},
                   compile=True)



from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
with open('model_summary.txt','w') as f:
    f.write(str(model.summary()))

################ Evaluation ####################################

def evaluation(true, predict, domain):
    true_label = []
    predict_label = []

    if domain == 'restaurant':

        for line in predict:
            predict_label.append(line.strip())

        for line in true:
            true_label.append(line.strip())

        print(classification_report(true_label, predict_label,
                                    ['Food', 'Staff', 'Ambience', 'Anecdotes', 'Price', 'Miscellaneous'], digits=3))

    elif domain == 'drugs_cadec':
        for line in predict:
            predict_label.append(line.strip())

        for line in true:
            true_label.append(line.strip())

        print(classification_report(true_label, predict_label, digits=3))

    else:
        for line in predict:
            label = line.strip()
            if label == 'smell' or label == 'taste':
                label = 'taste+smell'
            predict_label.append(label)

        for line in true:
            label = line.strip()
            if label == 'smell' or label == 'taste':
                label = 'taste+smell'
            true_label.append(label)

        #print(classification_report(true_label, predict_label,
        #                            ['feel', 'taste+smell', 'look', 'overall', 'None'], digits=3))
        print(classification_report(true_label, predict_label, digits=3))
        #cls_dict=classification_report(true_label, predict_label, digits=3,output_dict=True)
        #print(pd.DataFrame.from_dict(cls_dict))
        with open('classification_report.txt','w') as r:
            r.write(classification_report(true_label, predict_label, digits=3))

        #labels to label the confusion matrix
        labels=['live', 'safety', 'price', 'quiet', 'dining', 'nightlife', 'transit-location', 'touristy', 'shopping', 'green-culture', 'multicultural', 'general']
        cm=confusion_matrix(true_label, predict_label,labels=labels)
        print(pd.DataFrame(cm, index=labels, columns=labels))

        #print micro f1 score
        from sklearn.metrics import f1_score
        print('Micro F1 score\n',f1_score(true_label, predict_label, average='micro'))
        #print('Weighted macro F1 score\n',f1_score(true_label, predict_label, average='weighted'))


def prediction(test_labels, aspect_probs, cluster_map, domain):
    label_ids = np.argsort(aspect_probs, axis=1)[:, -1]
    predict_labels = [cluster_map[label_id] for label_id in label_ids]
    evaluation(open(test_labels), predict_labels, domain)


## Create a dictionary that map word index to word 
vocab_inv = {}
for w, ind in vocab.items():
    vocab_inv[ind] = w



test_fn = K.function([model.get_layer('sen_input_emb').input, model.get_layer('sentence_input').input, model.get_layer('sentence_average').input, K.learning_phase()],
                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])
att_weights, aspect_probs = [], []



import os
from w2vEmbReader import W2VEmbReader as EmbReader
emb_name=args.embname
emb_reader = EmbReader(os.path.join("..", "preprocessed_data"), emb_name)
from sememe_expansion_cls import Sememe
import numpy as np



for batch in tqdm(test_x):
    sem = Sememe(vocab, batch, emb_reader)
    input_sememe, input_average, input_embedding = sem.run_script()
    cur_att_weights, cur_aspect_probs = test_fn([input_embedding,input_sememe,input_average,0])
    print('current shapes:  ', np.array(cur_att_weights).shape,np.array(cur_aspect_probs).shape)
    att_weights.append(cur_att_weights)
    aspect_probs.append(cur_aspect_probs)
print('after appending   ',np.array(att_weights).shape,np.array(aspect_probs).shape)



att_weights = np.concatenate(att_weights)
aspect_probs = np.concatenate(aspect_probs)
print('after concat  ',np.array(att_weights).shape,np.array(aspect_probs).shape)


import pickle
with open('aspect_probs.pkl','wb') as f:
    pickle.dump(aspect_probs, f)



######### Topic weight ###################################

topic_weight_out = open(out_dir + '/topic_weights', 'wt', encoding='utf-8')
labels_out = open(out_dir + '/labels.txt', 'wt', encoding='utf-8')
print('Saving topic weights on test sentences...')
for probs in aspect_probs:
    labels_out.write(str(np.argmax(probs)) + "\n")
    weights_for_sentence = ""
    for p in probs:
        weights_for_sentence += str(p) + "\t"
    weights_for_sentence.strip()
    topic_weight_out.write(weights_for_sentence + "\n")


## Save attention weights on test sentences into a file
att_out = open(out_dir + '/att_weights', 'wt', encoding='utf-8')
print('Saving attention weights on test sentences...')
test_x = np.concatenate(test_x)
for c in range(len(test_x)):
    att_out.write('----------------------------------------\n')
    att_out.write(str(c) + '\n')

    word_inds = [i for i in test_x[c] if i != 0]
    line_len = len(word_inds)
    weights = att_weights[c]
    weights = weights[(overall_maxlen - line_len):]

    words = [vocab_inv[i] for i in word_inds]
    att_out.write(' '.join(words) + '\n')
    for j in range(len(words)):
        att_out.write(words[j] + ' ' + str(round(weights[j], 3)) + '\n')

######################################################
# Uncomment the below part for F scores
######################################################

## cluster_map need to be specified manually according to the top words in each inferred aspect (save in aspect.log)
    
# map for the pre-trained restaurant model (under pre_trained_model/restaurant)
# cluster_map = {0: 'Live', 1: 'touristy', 2: 'price', 3: 'price',
#            4: 'price', 5: 'price', 6:'price',  7: 'multicultural', 8: 'green-culture', 
#            9: 'Food', 10: 'Food', 11: 'Anecdotes', 
#            12: 'Ambience', 13: 'Staff'}
 
cluster_map={0: 'general', 1: 'general', 2: 'touristy', 3:'general', 4: 'safety', 5:'general', 6: 'price', 
7: 'general', 8: 'transit-location', 9:'general', 10: 'price', 11: 'general', 12: 'shopping', 13: 'general', 14: 'general', 15: 'multicultural',
16: 'multicultural', 17: 'touristy', 18: 'safety', 19: 'general', 20: 'nightlife', 21:'live', 22: 'general', 23: 'dining', 24:'general', 25: 'general', 
26: 'multicultural', 27: 'touristy', 28: 'price', 29: 'price', 30: 'general', 31: 'general', 32: 'general', 33: 'general', 34: 'general', 35: 'general',
36: 'price', 37: 'dining', 38: 'general', 39: 'general', 40: 'price', 41: 'safety ', 42: 'touristy', 43: 'price', 44: 'general', 45: 'general',
46: 'general', 47: 'touristy', 48: 'general', 49: 'general'}

#print '--- Results on %s domain ---' % (args.domain)
#test_labels = r'..\preprocessed_data\sentihood\\test_label.txt'
test_labels = r'../preprocessed_data/sentihood/test_label.txt'
#test_labels = '../preprocessed_data/sentihood/test_labels_filtered.txt' #changing test_label file to test_labels_filtered : SD
prediction(test_labels, aspect_probs, cluster_map, domain=args.domain)