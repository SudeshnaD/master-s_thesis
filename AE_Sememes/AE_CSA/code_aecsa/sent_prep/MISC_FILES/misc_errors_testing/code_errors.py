import keras.backend as K
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from keras import constraints
import numpy as np
import theano.tensor as T


class Attention(Layer):
    def __init__(self, W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Content Attention mechanism.
        Supports Masking.
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
            assert type(input_shape) == list
            assert len(input_shape) == 2

            self.steps = input_shape[0][1]

            self.W = self.add_weight((input_shape[0][-1], input_shape[1][-1]),
                                        initializer=self.init,
                                        #name='{}_W'.format(self.name),
                                        regularizer=self.W_regularizer,
                                        constraint=self.W_constraint)
            if self.bias:
                self.b = self.add_weight((1,),
                                        initializer='zero',
                                        #name='{}_b'.format(self.name),
                                        regularizer=self.b_regularizer,
                                        constraint=self.b_constraint)
            self.built = True