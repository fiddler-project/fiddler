from __future__ import division
import numpy as np
import tensorflow as tf
import sys


class RNN(object):

    def __init__(self, producer, data, batch_size, lstm_size, num_layers=2, num_steps=30,
                 num_classes=10000, num_epochs=50, learning_rate=1e-4):
        self.producer, self.data = producer, data
        self.batch_size = batch_size
        # Dimensions for each RNN cell's parameters (i.e. c and h)
        self.lstm_size = lstm_size

        self.num_layers = num_layers

        # number of time steps for training (aka unrolling!)
        self.num_steps = num_steps

        # number of classes to predict
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.x = tf.placeholder(tf.int32, shape=(
            self.batch_size, self.num_steps))
        self.y = tf.placeholder(tf.int32, shape=(
            self.batch_size, self.num_steps))

        # The second value for dimension depicts separate states for c and h
        # of a cell in RNN. Therefore, it is hardcoded as 2.
        self.init_state = tf.placeholder(
            tf.float32, [self.num_layers, 2, self.batch_size, self.lstm_size])

    def _build(self):
        """ Build the network """
        embedding_params = tf.get_variable(
            'embedding_matrix', [self.num_classes, self.lstm_size])
        rnn_input = tf.nn.embedding_lookup(embedding_params, self.x)

        # Each tensor will be of shape (2, self.batch_size, self.lstm_size)
        # since axis = 0
        state_per_layer = tf.unstack(self.init_state, axis=0)

        # Crate LSTM state tuples for each layer using `state_per_layer`
        rnn_states = tuple(
            [tf.contrib.rnn.LSTMStateTuple(state_per_layer[i][0], state_per_layer[i][1])
             for i in range(self.num_layers)]
        )

        single_cell = tf.nn.rnn_cell.LSTMCell(self.lstm_size, forget_bias=1.0)
        multi_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell for _ in xrange(self.num_layers)],
                                                 state_is_tuple=True)

        # Update `state` after each training step
        output, self.state = tf.nn.dynamic_rnn(multi_cell, rnn_input, dtype=tf.float32,
                                               initial_state=rnn_states)

        # Output from lstm has shape (2, self.batch_size, self.lstm_size)
        # In order to use it for softmax layer, we have to reshape it!
        output = tf.reshape(output, (-1, self.lstm_size))
        self._add_softmax_layer(output)

    def _add_softmax_layer(self, output):
        w_softmax = tf.Variable(tf.random_uniform(
            (self.lstm_size, self.num_classes), -1, 1))
        b_softmax = tf.Variable(tf.random_uniform([self.num_classes], -1, 1))
        class_probs = tf.nn.xw_plus_b(output, w_softmax, b_softmax)
        labels = tf.reshape(self.y, [-1])  # Flatten `self.y`
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_probs, labels=labels))
        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

    def train(self):
        # build network
        self._build()

        # train
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            state = np.zeros(
                (self.num_layers, 2, self.batch_size, self.lstm_size))
            for epoch in xrange(self.num_epochs):
                losses = 0.0
                print "Training: Epoch {}".format(epoch)
                for step, data in enumerate(self.producer(self.data, self.batch_size, self.num_steps)):
                    x, y = data
                    loss, _, state = sess.run([self.loss, self.train_step, self.state],
                                              feed_dict={self.x: x,
                                                         self.y: y,
                                                         self.init_state: state})
                    losses += loss
                    sys.stdout.write(" Steps {} \r".format(step))
                    sys.stdout.flush()
                print "Avg. loss for Epoch {}: {}".format(epoch, losses / (step + 1))
            saver.save(sess, "models/lstm-final")
