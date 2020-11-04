import logging
import os
import pickle

import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input, Lambda, LSTM, Concatenate
from keras.models import Model
from keras.constraints import MaxNorm

from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin, SememeAttention
from Embedding_Ext_layer import Embedding_Ext


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def create_model(args, maxlen, vocab, emb_reader):
    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = K.l2_normalize(weight_matrix, axis=-1)
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(w_n.shape[0].eval())))
        return args.ortho_reg * reg

    vocab_size = len(vocab)


    if args.embname:
        from w2vEmbReader import W2VEmbReader as EmbReader
        #emb_reader = EmbReader(os.path.join("..", "preprocessed_data", args.domain), args.embname)
        emb_reader = EmbReader(os.path.join("..", "preprocessed_data"), args.embname)
        aspect_matrix = emb_reader.get_aspect_matrix(args.aspect_size)
        args.aspect_size = emb_reader.aspect_size
        args.emb_dim = emb_reader.emb_dim



    ##### Inputs #####
    sen_input_emb = Input(shape=(maxlen,args.emb_dim), dtype='float32', name='sen_input_emb')
    sentence_input = Input(shape=(maxlen,args.emb_dim), dtype='float32', name='sentence_input')
    sentence_average = Input(shape=(args.emb_dim,), dtype='float32', name='sentence_average')
    neg_input = Input(shape=(args.neg_size, args.emb_dim), dtype='float32', name='neg_input')
    #sentence_average=K.reshape(sentence_average,(100,1))


    # def emb_extraction(sen_input_default):
    #     semb = Embedding_Ext(vocab,sen_input_default,emb_reader)
    #     sen_emb_default = semb.run_script()
    #     return sen_emb_default
    # sen_inpd_emb=Lambda(lambda x: emb_extraction(x), name='lambda_layer')(sen_input_default)
    # lamda_stop_grad = Lambda(lambda x: K.stop_gradient(x))(sen_inpd_emb)


    sen_hr=LSTM(500, return_state=True)(sen_input_emb)


    # ##### Construct word embedding layer #####
    # word_emb = Embedding(vocab_size, args.emb_dim,
    #                       mask_zero=True, name='word_emb',
    #                       embeddings_constraint=MaxNorm(10))


    ##### Compute sentence representation #####
    #e_w = word_emb(sentence_input)
    #y_s = Average(name='average_1')(e_w)
    att_weights = SememeAttention(name='att_weights',  #conditioned on the embedding of the word ewi as well as the global context of the sentence
                             W_constraint=MaxNorm(10),
                             b_constraint=MaxNorm(10))([sentence_input, sentence_average])



    z_s = WeightedSum(name='weightedsum')([sentence_input, att_weights]) #sentence embedding==weighted summation of word embedding



    add_sen_rep=Concatenate(axis=1)([z_s,sen_hr[0]])
    join_rep=Dense(args.emb_dim, name='dense_0')(add_sen_rep)
    join_rep=Lambda(lambda x: K.tanh(x))(join_rep)


    ##### Compute representations of negative instances #####
    #e_neg = word_emb(neg_input)
    #z_n = Average()(e_neg)


    #z_n = neg_input
    #z_n = K.cast(z_n, K.floatx())


    ##### Reconstruction #####
    p_t = Dense(args.aspect_size, name='dense')(join_rep) #z_s==weighted summation of word embedding
    p_t = Activation('softmax', name='p_t')(p_t)
    r_s = WeightedAspectEmb(args.aspect_size, args.emb_dim, name='aspect_emb',
                            W_constraint=MaxNorm(10),
                            W_regularizer=ortho_reg)(p_t)



    ##### Loss #####
    loss = MaxMargin(name='max_margin')([z_s, neg_input, r_s])

    

    
    model = Model(inputs=[sen_input_emb, sentence_input, sentence_average, neg_input], outputs=[loss])




    ### Word embedding and aspect embedding initialization ######
    if args.embname:
        from w2vEmbReader import W2VEmbReader as EmbReader
        #logger.info('Initializing word embedding matrix')
        #embs = model.get_layer('word_emb').embeddings
        #K.set_value(embs, emb_reader.get_emb_matrix_given_vocab(vocab, K.get_value(embs)))
        logger.info('Initializing aspect embedding matrix as centroid of kmean clusters')
        K.set_value(model.get_layer('aspect_emb').W, aspect_matrix)

    return model


""" if __name__=='__main__':

    with open('vocab_c.pkl','rb') as f:
            vocab_c=pickle.load(f)
    vocab = vocab_c

    import utils as U
    parser = U.add_common_args()
    parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=100,
                        help="Embeddings dimension (default=100)")
    parser.add_argument("-as", "--aspect-size", dest="aspect_size", type=int, metavar='<int>', default=14,
                        help="The number of aspects specified by users (default=14)")
    parser.add_argument("--emb-name",  type=str,
                        help="The name to the word embeddings file", default="w2v_64k_unigram_100d.model")
    parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=15,
                        help="Number of epochs (default=15)")
    parser.add_argument("-n", "--neg-size", dest="neg_size", type=int, metavar='<int>', default=20,
                        help="Number of negative instances (default=20)")
    parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234,
                        help="Random seed (default=1234)")
    parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='adam',
                        help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=adam)")
    parser.add_argument("--ortho-reg", dest="ortho_reg", type=float, metavar='<float>', default=0.1,
                        help="The weight of orthogonal regularization (default=0.1)")
    args = parser.parse_args()


    model = create_model(args, 69, vocab) """