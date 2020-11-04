import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import classification_report,precision_score,confusion_matrix
import keras.backend as K
from keras.preprocessing import sequence

import utils as U
import reader as dataset
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin

######### Get hyper-params in order to rebuild the model architecture ###########
# The hyper parameters should be exactly the same as those used for training

parser = U.add_common_args()
parser.add_argument('-as','--aspect_size',type=int ,help='number of aspects')
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



print(len(train_x),len(test_x))
print(train_x[0])
print(test_x[0])
print(overall_maxlen)
test_x = sequence.pad_sequences(test_x, maxlen=overall_maxlen)
print('shape of text_x after padding: ',test_x.shape,len(test_x))
test_length = test_x.shape[0]
print('test_length %d' % test_length)



splits = []
for i in range(1, test_length // args.batch_size):
    splits.append(args.batch_size * i)
if test_length % args.batch_size:
    splits += [(test_length // args.batch_size) * args.batch_size]
test_x = np.split(test_x, splits)
print('len(splits) %d test_x %d' % (len(splits),len(test_x)))





############# Build model architecture, same as the model used for training #########

## Load the save model parameters
model = load_model(out_dir + '/model_param{}'.format(args.aspect_size),
                   custom_objects={"Attention": Attention, "Average": Average, "WeightedSum": WeightedSum,
                                   "MaxMargin": MaxMargin, "WeightedAspectEmb": WeightedAspectEmb,
                                   "max_margin_loss": U.max_margin_loss},
                   compile=True)



from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


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
        with open(out_dir+'/classification_report'+str(args.aspect_size)+'.txt','w') as r:
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



test_fn = K.function([model.get_layer('sentence_input').input, K.learning_phase()],
                     [model.get_layer('att_weights').output, model.get_layer('p_t').output])
att_weights, aspect_probs = [], []
for batch in tqdm(test_x):
    cur_att_weights, cur_aspect_probs = test_fn([batch, 0])
    print(np.array(cur_att_weights).shape, np.array(cur_aspect_probs).shape)
    att_weights.append(cur_att_weights)
    aspect_probs.append(cur_aspect_probs)
print(np.array(att_weights).shape, np.array(aspect_probs).shape)


att_weights = np.concatenate(att_weights)
aspect_probs = np.concatenate(aspect_probs)
print('after concatenation: ', np.array(att_weights).shape, np.array(aspect_probs).shape)




######### Topic weight ###################################

topic_weight_out = open(out_dir + '/topic_weights', 'wt', encoding='utf-8')
labels_out = open(out_dir + '/labels.txt', 'wt', encoding='utf-8')
print('Saving topic weights on test sentences...')
print(aspect_probs[0])
for probs in aspect_probs:
    labels_out.write(str(np.argmax(probs)) + "\n")
    weights_for_sentence = ""
    for p in probs:
        weights_for_sentence += str(p) + "\t"
    weights_for_sentence.strip()
    topic_weight_out.write(weights_for_sentence + "\n")
print(aspect_probs)


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
 
from output_dir.aspects_models_glovewiki100d.ClusterMaps import cluster_dict

cluster_map=cluster_dict[args.aspect_size]

#print '--- Results on %s domain ---' % (args.domain)
test_labels = '../preprocessed_data/sentihood/test_label.txt'
#test_labels = '../preprocessed_data/sentihood/test_labels_filtered.txt' #changing test_label file to test_labels_filtered : SD
prediction(test_labels, aspect_probs, cluster_map, domain=args.domain)