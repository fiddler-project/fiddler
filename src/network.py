from __future__ import division
import numpy as np
import tensorflow as tf
import sys


class RNN(object):

    def __init__(self, data, lstm_size, num_layers=2, num_epochs=50, learning_rate=1e-4):
        """
        `data` is dataset.Dataset object
        `lstm_size` is the Dimensions for each RNN cell's parameters (i.e. c and h)
        `num_layers` is number of layers in the network
        `num_epochs` is total number of epochs training is to be run for
        `learning_rate` is learning rate for gradient optimizer
        """
        self.data = data
        self.batch_size = self.data.batch_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = self.data.num_steps
        self.num_classes = self.data.vocab_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.x = tf.placeholder(tf.int32, shape=(
            None, None))
        self.y = tf.placeholder(tf.int32, shape=(
            None, None))

        # The second value for dimension depicts separate states for c and h
        # of a cell in RNN. Therefore, it is hardcoded as 2.
        self.init_state = tf.placeholder(
            tf.float32, [self.num_layers, 2, None, self.lstm_size])
        self.final_state = None

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
        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(
            tf.reshape(class_probs, [-1, self.num_classes]))
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, test_output=True):
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
                for step, data in enumerate(self.data.batch()):
                    x, y = data
                    loss, _, state = sess.run([self.loss, self.train_step, self.state],
                                              feed_dict={self.x: x,
                                                         self.y: y,
                                                         self.init_state: state})
                    losses += loss
                    # sys.stdout.write(" Steps {} \r".format(step))
                    # sys.stdout.flush()
                print "Avg. loss for Epoch {}: {}".format(
                    epoch, losses / (step + 1))
                if test_output:
                    print "---------- Generated text -----------"
                    print self.gen_text(sess)

            saver.save(sess, "models/lstm-final")

    def gen_text(self, sess, seed_input=None, size=500):
        text = ""
        if not seed_input:
            seed_input = self.data.idx_to_vocab[
                np.random.randint(0, self.data.vocab_size - 1)]
        test_input = [self.data.vocab_to_idx[c] for c in seed_input]
        for i in range(len(test_input)):
            x = np.array([test_input[i]])
            x = x.reshape((1, 1))
            out = self.predict_(x, sess, i == 0)[0]

        for i in range(size):
            element = np.random.choice(range(self.data.vocab_size), p=out)
            text += self.data.idx_to_vocab[element]
            x = np.array([element])
            x = x.reshape((1, 1))
            out = self.predict_(x, sess, False)[0]
        return text

    def predict_(self, x, sess, init_zero_state=True):
        if init_zero_state:
            print self.num_layers
            init_value = np.zeros(
                (self.num_layers, 2, 1, self.lstm_size))
        else:
            init_value = self.final_state
        out, state = sess.run(
            [self.softmax_out, self.state],
            feed_dict={
                self.x: x,
                self.init_state: init_value
            }
        )
        self.final_state = state
        return out