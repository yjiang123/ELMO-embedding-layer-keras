import keras.layers as layers
import tensorflow_hub as hub
from keras import backend as K

class ElmoEmbeddingLayer(Layer):

def __init__(self, mask, **kwargs):
    self.dimensions = 1024
    self.trainable = True
    self.mask = mask
    super(ElmoEmbeddingLayer, self).__init__(**kwargs)

def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))
        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

def call(self, inputs, mask=None):
        # inputs.shape = [batch_size, seq_len]
        seq_len = [inputs.shape[1]] * inputs.shape[0] # this will give a list of seq_len: [seq_len, seq_len, ..., seq_len] just like the official example.
        result = self.elmo(inputs={"tokens": K.cast(inputs, dtype=tf.string),
                                   "sequence_len": seq_len},
                      as_dict=True,
                      signature='tokens',
                      )['elmo']
     
        return result

def compute_mask(self, inputs, mask=None):
        if not self.mask:
            return None

        output_mask = K.not_equal(inputs, '--PAD--')
        return output_mask
def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.dimensions)`
