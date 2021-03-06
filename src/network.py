from __future__ import division
import numpy as np
import tensorflow as tf
import sys
import pickle

MODELS_PATH = "models/"


class RNN(object):

    def __init__(self, data, cell, cell_size, dropout=0.8, num_layers=2, num_epochs=50, learning_rate=1e-3, training=True):
        """
        `data` is dataset.Dataset object
        `cell_size` is the Dimensions for each RNN cell's parameters (i.e. c and h)
        `num_layers` is number of layers in the network
        `num_epochs` is total number of epochs training is to be run for
        `learning_rate` is learning rate for gradient optimizer
        """
        self.data = data
        self.batch_size = self.data.batch_size
        self.cell = cell
        self.cell_size = cell_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_steps = self.data.num_steps
        self.num_classes = self.data.vocab_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.training = training

        self.x = tf.placeholder(tf.int32, shape=(
            None, None), name="x")
        self.y = tf.placeholder(tf.int32, shape=(
            None, None), name="y")

        self.final_state = None
        # The second value for dimension depicts separate states for c and h
        # of a LSTM cell in RNN. Therefore, it is hardcoded as 2.
        # GRU cell in RNN has a single state
        if self.cell == "lstm":
            self.init_state = tf.placeholder(
                tf.float32, [self.num_layers, 2, None, self.cell_size], name="cell_state")
        else:
            self.init_state = tf.placeholder(
                tf.float32, [self.num_layers, None, self.cell_size], name="cell_state")
        self._build()

    def _build(self):
        """ Build the network """
        self.embedding_params = tf.get_variable(
            'embedding_matrix', [self.num_classes, self.cell_size])
        rnn_input = tf.nn.embedding_lookup(self.embedding_params, self.x)

        # Each tensor will be of shape (2, self.batch_size, self.cell_size) - LSTM
        # Each tensor will be of shape (self.batch_size, self.cell_size) - GRU
        # since axis = 0
        state_per_layer = tf.unstack(self.init_state, axis=0)

        # Create the RNN cells
        if self.cell == "lstm":
            rnn_states = tuple(
                [tf.contrib.rnn.LSTMStateTuple(state_per_layer[i][0], state_per_layer[i][1])
                 for i in range(self.num_layers)])
            single_cell = tf.nn.rnn_cell.LSTMCell(
                self.cell_size, forget_bias=1.0)
            # Use dropout only for training
            if self.dropout and self.training:
                single_cell = tf.contrib.rnn.DropoutWrapper(
                    single_cell, output_keep_prob=self.dropout)
            multi_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell for _ in xrange(self.num_layers)],
                                                     state_is_tuple=True)
        else:
            rnn_states = tuple([state_per_layer[i]
                                for i in range(self.num_layers)])
            single_cell = tf.nn.rnn_cell.GRUCell(self.cell_size)
            # Use dropout only for training
            if self.dropout and self.training:
                single_cell = tf.contrib.rnn.DropoutWrapper(
                    single_cell, output_keep_prob=self.dropout)
            multi_cell = tf.nn.rnn_cell.MultiRNNCell(
                [single_cell] * self.num_layers)

        # Update `state` after each training step
        output, self.state = tf.nn.dynamic_rnn(multi_cell, rnn_input, dtype=tf.float32,
                                               initial_state=rnn_states)
        # Output from lstm has shape (2, self.batch_size, self.cell_size)
        # In order to use it for softmax layer, we have to reshape it!
        output = tf.reshape(output, (-1, self.cell_size))
        self._add_softmax_layer(output)

    def _add_softmax_layer(self, output):
        self.w_softmax = tf.get_variable("w_softmax", initializer=tf.random_uniform(
            (self.cell_size, self.num_classes), -1, 1))
        self.b_softmax = tf.get_variable("b_softmax",
                                         initializer=tf.random_uniform([self.num_classes], -1, 1))
        class_probs = tf.nn.xw_plus_b(output, self.w_softmax, self.b_softmax)
        labels = tf.reshape(self.y, [-1])  # Flatten `self.y`
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_probs, labels=labels))
        # get the prediction accuracy
        self.softmax_out = tf.nn.softmax(
            tf.reshape(class_probs, [-1, self.num_classes]), name="softmax_out")
        self.predict = tf.cast(tf.argmax(self.softmax_out, axis=1), tf.int32)
        correct_prediction = tf.equal(self.predict, labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

    def train(self, save=False, model_name="rnn", test_output=True, test_seed=None, with_delim=True):
        losses_ = []
        train_accuracy = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if self.cell == "lstm":
                state = np.zeros(
                    (self.num_layers, 2, self.batch_size, self.cell_size))
            else:
                state = np.zeros(
                    (self.num_layers, self.batch_size, self.cell_size))
            for epoch in xrange(self.num_epochs):
                losses = accuracies = 0.0
                print "Training: Epoch {}".format(epoch)
                for step, data in enumerate(self.data.batch()):
                    x, y, epoch_size = data
                    loss, _, state, accuracy = sess.run([self.loss, self.train_step, self.state, self.accuracy],
                                                        feed_dict={self.x: x,
                                                                   self.y: y,
                                                                   self.init_state: state})
                    losses += loss
                    accuracies += accuracy
                    losses_.append(loss)
                    train_accuracy.append(accuracy)
                    sys.stdout.write(
                        " Steps {}/{} \r".format(step, epoch_size))
                    sys.stdout.flush()
                    self.finish_state = state
                print "Avg. loss for Epoch {}: {}".format(
                    epoch, losses / (step + 1))
                print "Avg. accuracy for epoch {}: {}".format(
                    epoch, accuracies / (step + 1))
                if test_output:
                    print "---------- Generated text -----------"
                    print self.gen_text(
                        sess, seed_input=test_seed, with_delim=with_delim)
            if save:
                saver.save(sess, "{}{}/".format(MODELS_PATH, model_name),
                           write_meta_graph=True)
                test_accuracy = self.test(sess)
                self._write_data(model_name, train_accuracy,
                                 test_accuracy, losses_)

    def _write_data(self, model_name, train_accuracy, test_accuracy, losses):
        string = []
        for i in xrange(len(train_accuracy)):
            string += ["{},{},1\n".format(train_accuracy[i], losses[i])]
        string += ["{},0,0\n".format(test_accuracy)]
        with open("{}{}/train_stats.txt".format(MODELS_PATH, model_name), "w") as f:
            f.write(u''.join(string))

    def test(self, sess):
        self.training = False
        accuracy = 0
        for step, data in enumerate(self.data.batch(train_data=False)):
            x, y, _ = data
            accuracy += sess.run([self.accuracy], feed_dict={
                self.x: x,
                self.y: y,
                self.init_state: self.finish_state
            })[0]
        print "Test accuracy: {}".format(accuracy / (step + 1))
        self.training = True
        return accuracy / (step + 1)

    def gen_text(self, sess, model_path=None, seed_input=None, with_delim=True, size=300):
        self.training = False
        if not sess:
            sess = tf.Session()
            new_saver = tf.train.Saver()
            new_saver.restore(sess, model_path)
        if with_delim:
            text, prev_char = [], ""
            char_in = self.data.vocab_to_idx["<s>"]
            out = self.predict_(np.array([char_in]).reshape((1, 1)), sess)[0]
            while prev_char != "</s>":
                text.append(prev_char)
                char_in = np.random.choice(range(self.data.vocab_size), p=out)
                prev_char = self.data.idx_to_vocab[char_in]
                out = self.predict_(
                    np.array([char_in]).reshape((1, 1)), sess, False)[0]
            t = u''.join(text).encode('utf-8').strip()
        else:
            t = self._gen_text_from_seed(sess, seed_input, size)
        self.training = True
        return t

    def _gen_text_from_seed(self, sess, seed_input, size):
        if not seed_input:
            seed_input = self.data.idx_to_vocab[
                np.random.randint(0, self.data.vocab_size - 1)]
        text = seed_input
        test_input = [self.data.vocab_to_idx[c] for c in seed_input]
        for i in range(len(test_input)):
            x = np.array([test_input[i]]).reshape((1, 1))
            out = self.predict_(x, sess, i == 0)[0]
        gen = [text]
        for i in range(size):
            element = np.random.choice(range(self.data.vocab_size), p=out)
            gen.append(self.data.idx_to_vocab[element])
            x = np.array([element]).reshape((1, 1))
            out = self.predict_(x, sess, False)[0]
        return u''.join(gen).encode('utf-8').strip()

    def predict_(self, x, sess, init_zero_state=True):
        if not init_zero_state:
            init_value = self.final_state
        elif self.cell == "lstm":
            init_value = np.zeros((self.num_layers, 2, len(x), self.cell_size))
        else:
            init_value = np.zeros((self.num_layers, len(x), self.cell_size))
        out, state = sess.run(
            [self.softmax_out, self.state],
            feed_dict={
                self.x: x,
                self.init_state: init_value
            }
        )
        self.final_state = state
        return out

    def _build_model_config(self):
        return dict(data=self.data,
                    num_layers=self.num_layers,
                    cell=self.cell,
                    cell_size=self.cell_size,
                    dropout=self.dropout,
                    learning_rate=self.learning_rate)

    def save(self, model_name):
        with open("{}{}/rnn.pickle".format(MODELS_PATH, model_name), "w") as f:
            model_config = self._build_model_config()
            pickle.dump(model_config, f)
