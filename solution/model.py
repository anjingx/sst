import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.seq2seq.python.ops.loss import sequence_loss
from tensorflow.contrib.lookup.lookup_ops import MutableHashTable
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.session_bundle import exporter

from rnn_cell import GRUCell, BasicLSTMCell, MultiRNNCell, BasicRNNCell

PAD_ID = 0
UNK_ID = 1
_START_VOCAB = ['_PAD', '_UNK']

class RNN(object):
    def __init__(self,
            num_symbols,
            num_embed_units,
            num_units,
            num_layers,
            num_labels,
            embed,
            learning_rate=0.005,
            max_gradient_norm=5.0):
        
        self.texts = tf.placeholder(tf.string, (None, None), 'texts')  # batch*len
        self.texts_length = tf.placeholder(tf.int32, (None), 'texts_length')  # batch
        self.labels = tf.placeholder(tf.int64, (None), 'labels')  # batch
        
        self.symbol2index = MutableHashTable(
                key_dtype=tf.string,
                value_dtype=tf.int64,
                default_value=UNK_ID,
                shared_name="in_table",
                name="in_table",
                checkpoint=True)
        # build the vocab table (string to index)
        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), 
                trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)


        self.index_input = self.symbol2index.lookup(self.texts)   # batch*len
        
        # build the embedding table (index to vector)
        if embed is None:
            # initialize the embedding randomly
            self.embed = tf.get_variable('embed', [num_symbols, num_embed_units], tf.float32)
        else:
            # initialize the embedding by pre-trained word vectors
            self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
        
        self.embed_input = tf.nn.embedding_lookup(self.embed, self.index_input) #batch*len*embed_unit

        cell = MultiRNNCell([BasicLSTMCell(num_units) for _ in range(num_layers)])
        
        outputs, states = dynamic_rnn(cell, self.embed_input, 
                self.texts_length, dtype=tf.float32, scope="rnn")

        vectors = states[-1][-1]

        with tf.variable_scope('logits'):
            weight = tf.get_variable("weights", [num_units, num_labels])
            bias = tf.get_variable("biases", [num_labels])
            logits = tf.matmul(vectors, weight) + bias

        self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits), name='loss')
        predict_labels = tf.argmax(logits, 1, 'predict_labels')
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.labels, predict_labels), tf.int32), name='accuracy')

        self.params = tf.trainable_variables()
            
        # calculate the gradient of parameters
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)
        
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=5, pad_step_number=True)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))
    
    def train_step(self, session, data):
        input_feed = {self.texts: data['texts'],
                self.texts_length: data['texts_length'],
                self.labels: data['labels']}
        output_feed = [self.loss, self.accuracy, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)
