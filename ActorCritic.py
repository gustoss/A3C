import numpy as np
import math
import tensorflow as tf
import random as rd


def _conv_layer(x, shape, stride, activation_fn=None, name='conv_layer'):
    with tf.variable_scope(name):
        nin = shape[0] * shape[1] * shape[2]
        nout = shape[0] * shape[1] * shape[3]
        maxval = math.sqrt(6 / (nin + nout))
        w = tf.get_variable('weight', shape=shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-maxval, maxval=maxval))
        b = tf.get_variable('bias', shape=[shape[3]], dtype=tf.float32, initializer=tf.constant_initializer(0))
        conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='VALID', name='conv')
        conv = tf.nn.bias_add(conv, b)
        return activation_fn(conv, name='activation') if activation_fn is not None else conv

def _max_layer(x, ksize, stride, activation_fn=None, name='max_pool_layer'):
    m = tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1], padding='VALID', name=name)
    return activation_fn(m, name='activation') if activation_fn is not None else m

def _fully_layer(x, shape, activation_fn=None, name='fully_layer'):
    with tf.variable_scope(name):
        maxval = math.sqrt(6 / (shape[0] + shape[1]))
        w = tf.get_variable('weight', shape=shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-maxval, maxval=maxval))
        b = tf.get_variable('bias', shape=[shape[1]], dtype=tf.float32, initializer=tf.constant_initializer(.00001))
        fully = tf.matmul(x, w) + b
        return activation_fn(fully) if activation_fn is not None else fully

def _out_conv2d(value, filter, strides):
    return int(float(value - filter + 1) / float(strides))

class AC:
    def __init__(self, inpu, output, sess):
        self.input = inpu
        self.output = output
        self.sess = sess

        self.lstm_hidden = 128

        self.x = tf.placeholder(tf.float32, self.input, name='x')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('prediction'):
            h_conv_1 = _conv_layer(self.x, [5, 5, 1, 32], 1, name='conv_1')
            h_max_1 = _max_layer(h_conv_1, 3, 2, activation_fn=tf.nn.relu, name='max_pool_1')

            h_conv_2 = _conv_layer(h_max_1, [3, 3, 32, 32], 1, name='conv_2')
            h_max_2 = h_conv_2

            h_conv_3 = _conv_layer(h_max_2, [2, 2, 32, 32], 1, name='conv_3')
            h_max_3 = _max_layer(h_conv_3, 3, 2, activation_fn=tf.nn.relu, name='max_pool_3')

            wi = _out_conv2d(_out_conv2d(_out_conv2d(_out_conv2d(_out_conv2d(self.input[1], 5, 1), 3, 2), 3, 1), 2, 1), 3, 2)
            he = _out_conv2d(_out_conv2d(_out_conv2d(_out_conv2d(_out_conv2d(self.input[2], 5, 1), 3, 2), 3, 1), 2, 1), 3, 2)
            
            reshape = tf.reshape(h_max_3, [-1, wi * he * 32], name='reshape_conv')

            h_full_1 = _fully_layer(reshape, [wi * he * 32, 256], activation_fn=tf.nn.relu, name='fully_1')
            h_full_1_drop = tf.nn.dropout(h_full_1, keep_prob=self.keep_prob, name='dropout_1')

            # Adicionar o lstm
            self.lstm_state = (tf.placeholder(tf.float32, [1, self.lstm_hidden], name='cell_state'),
                               tf.placeholder(tf.float32, [1, self.lstm_hidden], name='hidden_state'))

            self.initial_lstm_state = (np.zeros([1, self.lstm_hidden], np.float32),
                                       np.zeros([1, self.lstm_hidden], np.float32))

            lstm_state = tf.contrib.rnn.LSTMStateTuple(*self.lstm_state)
            lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_hidden)
            batch = tf.shape(h_full_1_drop)[:1]
            lstm_input = tf.expand_dims(h_full_1_drop, [0])
            lstm_out, self.new_lstm_state = tf.nn.dynamic_rnn(lstm, lstm_input, batch, lstm_state)
            lstm_out = tf.squeeze(lstm_out, [0])
            lstm_out_drop = tf.nn.dropout(lstm_out, keep_prob=self.keep_prob, name='dropout_4')

        with tf.variable_scope('actor_prediction'):
            h_full_actor = _fully_layer(lstm_out_drop, [self.lstm_hidden, self.output], name='fully_actor')
            self.action_logits = h_full_actor
            self.action = tf.multinomial(self.action_logits, 1)
            self._action = tf.argmax(tf.nn.softmax(self.action_logits), axis=1)

        with tf.variable_scope('critic_prediction'):
            h_full_critic = _fully_layer(lstm_out_drop, [self.lstm_hidden, 1], name='fully_critic')
            self.value_critic = h_full_critic

    def act(self, x, lstm_state, rand=True):
        run = []
        feed_dict = {
            self.x: x,
            self.lstm_state: lstm_state,
            self.keep_prob: 1.
        }
        if rand:
            run = [
                self.action,
                self.new_lstm_state
            ]
            out, new_lstm = self.sess.run(run, feed_dict=feed_dict)
            out = out[0]
        else:
            run = [
                self._action,
                self.new_lstm_state
            ]
            out, new_lstm = self.sess.run(run, feed_dict=feed_dict)
        return out, new_lstm

    def initial_state(self):
        return self.initial_lstm_state

    def load(self, sess, save_path):
        with tf.variable_scope('global', reuse=tf.AUTO_REUSE):
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
            vars_save = {v.op.name: v for v in global_vars}
            saver = tf.train.Saver(vars_save)
            saver.restore(sess, save_path)
