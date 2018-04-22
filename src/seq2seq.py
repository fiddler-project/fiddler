import tensorflow as tf
import numpy as np

import numpy as np
import re
import io
import math


class Dataset(object):

    def __init__(self, path, batch_size=100, num_steps=5, with_delim=True, train_percent=0.8):
        """ `path` is processed file's path """
        print("Loading dataset...")
        with io.open(path, encoding='utf-8', mode='r') as f:
            raw_data = f.read()
            print("Data length:", len(raw_data))

        self.vocab = set(raw_data)
        start_symbol, end_symbol, go_symbol, end_seq, pad = '<s>', '</s>', '<GO>', '<EOS>', '<PAD>'
        if with_delim:
            self.vocab.update(
                {start_symbol, end_symbol, go_symbol, end_seq, pad})

        self.vocab_size = len(self.vocab)
        self.idx_to_vocab = dict(enumerate(self.vocab))
        self.vocab_to_idx = dict(
            zip(self.idx_to_vocab.values(), self.idx_to_vocab.keys()))
        print "GO Symbol is at:", self.vocab_to_idx['<GO>']
        tunes = map(lambda s: s.strip(), raw_data.split('\n\n'))
        self.data = []
        pad_seq = [self.vocab_to_idx[pad]] * (num_steps - 2)
        for t in tunes:
            self.data += [self.vocab_to_idx[start_symbol]] + \
                         [self.vocab_to_idx[c] for c in t] + \
                         [self.vocab_to_idx[end_symbol]]
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.train_limit = int(math.floor(len(self.data) * train_percent))

    def batch(self, train_data=True):
        """ Returns a new batch """
        if train_data:
            raw_data = np.array(self.data[:self.train_limit])
        else:
            raw_data = np.array(self.data[self.train_limit + 1:])
        raw_data = np.array(self.data)
        data_len = len(raw_data)
        rows = data_len // (self.num_steps)
        data = np.reshape(raw_data[0: rows * (self.num_steps)],
                          [rows, self.num_steps])
        x = data[::2, :][:, ::-1]
        y = data[1::2, :]

        epoch_size = ((rows / 2) // self.batch_size)

        for i in xrange(epoch_size):
            start_idx = i * self.batch_size
            remaining_rows = (rows // 2) - (i * self.batch_size)
            l = self.batch_size if remaining_rows >= self.batch_size else remaining_rows
            go_col = np.repeat([self.vocab_to_idx['<GO>']], l).reshape(l, 1)
            x_ret = x[start_idx: start_idx + self.batch_size, :]
            y_ret = np.hstack(
                (go_col, y[start_idx: start_idx + self.batch_size, :]))
            yield x_ret, y_ret, epoch_size


data_path = "data/abc_transposed.txt"
d = Dataset(data_path, num_steps=10)

# build TF graph
x_seq_length = 10
y_seq_length = 10
epochs = 1
batch_size = 100
nodes = 100
embed_size = 100
bidirectional = True

tf.reset_default_graph()
sess = tf.InteractiveSession()

# Tensor where we will feed the data into graph
inputs = tf.placeholder(tf.int32, (None, None), 'inputs')
outputs = tf.placeholder(tf.int32, (None, None), 'output')
targets = tf.placeholder(tf.int32, (None, None), 'targets')

# Embedding layers
input_embedding = tf.Variable(tf.random_uniform(
    (len(d.vocab_to_idx), embed_size), -1.0, 1.0), name='enc_embedding')
output_embedding = tf.Variable(tf.random_uniform(
    (len(d.vocab_to_idx), embed_size), -1.0, 1.0), name='dec_embedding')
date_input_embed = tf.nn.embedding_lookup(input_embedding, inputs)
date_output_embed = tf.nn.embedding_lookup(output_embedding, outputs)

with tf.variable_scope("encoding") as encoding_scope:

    if not bidirectional:

        # Regular approach with LSTM units
        lstm_enc = tf.contrib.rnn.LSTMCell(nodes)
        _, last_state = tf.nn.dynamic_rnn(
            lstm_enc, inputs=date_input_embed, dtype=tf.float32)

    else:

        # Using a bidirectional LSTM architecture instead
        enc_fw_cell = tf.contrib.rnn.LSTMCell(nodes)
        enc_bw_cell = tf.contrib.rnn.LSTMCell(nodes)

        ((enc_fw_out, enc_bw_out), (enc_fw_final, enc_bw_final)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_fw_cell,
                                                                                                   cell_bw=enc_bw_cell, inputs=date_input_embed, dtype=tf.float32)
        enc_fin_c = tf.concat((enc_fw_final.c, enc_bw_final.c), 1)
        enc_fin_h = tf.concat((enc_fw_final.h, enc_bw_final.h), 1)
        last_state = tf.contrib.rnn.LSTMStateTuple(c=enc_fin_c, h=enc_fin_h)


with tf.variable_scope("decoding") as decoding_scope:

    if not bidirectional:
        lstm_dec = tf.contrib.rnn.LSTMCell(nodes)
    else:
        lstm_dec = tf.contrib.rnn.LSTMCell(2*nodes)

    dec_outputs, _ = tf.nn.dynamic_rnn(
        lstm_dec, inputs=date_output_embed, initial_state=last_state)


logits = tf.layers.dense(dec_outputs, units=len(d.vocab_to_idx), use_bias=True)
y = tf.nn.softmax(logits)
# connect outputs to
with tf.name_scope("optimization"):
    # Loss function
    loss = tf.contrib.seq2seq.sequence_loss(
        logits, targets, tf.ones([batch_size, y_seq_length]))
    # Optimizer
    optimizer = tf.train.AdamOptimizer(1e-2).minimize(loss)

import time
import tqdm
sess.run(tf.global_variables_initializer())
epochs = 5
for epoch_i in range(epochs):
    start_time = time.time()
    for batch_i, (source_batch, target_batch, _) in tqdm.tqdm(enumerate(d.batch())):
        _, batch_loss, batch_logits = sess.run([optimizer, loss, logits],
                                               feed_dict={inputs: source_batch,
                                                          outputs: target_batch[:, :-1],
                                                          targets: target_batch[:, 1:]})
        dec_input = target_batch[:, :-1]
        accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:, 1:])
    print('Epoch {:3} Loss: {:>6.3f} Accuracy: {:>6.4f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,
                                                                                          accuracy, time.time() - start_time))

# Prediction
pad_seq = [d.vocab_to_idx['<PAD>']] * 3
warm_up = "M:6/8\n"
i = [d.vocab_to_idx['<s>']] + [d.vocab_to_idx[l] for l in warm_up]
source_batch = np.array([i])
dec_input = np.array([d.vocab_to_idx["<GO>"]]).reshape(1, 1)
s = []
prediction = [None]
while prediction[0] != d.vocab_to_idx['</s>']:
    batch_logits = sess.run(y,
                            feed_dict={inputs: source_batch,
                                       outputs: dec_input})
    prediction = np.array(
        [np.random.choice(range(d.vocab_size), p=batch_logits[:, -1][0])])
    s += [d.idx_to_vocab[prediction[0]]]
    dec_input = np.hstack([dec_input, prediction[:, None]])
print warm_up + "".join(s)
