import numpy as np


class Dataset(object):

    def __init__(self, path, batch_size=1000, num_steps=10):
        """ `path` is processed file's path """
        print "Loading dataset..."
        with open(path, 'r') as f:
            raw_data = f.read()
            print("Data length:", len(raw_data))
        self.vocab = set(raw_data)
        self.vocab_size = len(self.vocab)
        self.idx_to_vocab = dict(enumerate(self.vocab))
        self.vocab_to_idx = dict(
            zip(self.idx_to_vocab.values(), self.idx_to_vocab.keys()))
        self.data = [self.vocab_to_idx[c] for c in raw_data]
        del raw_data
        self.batch_size = batch_size
        self.num_steps = num_steps

    def batch(self):
        """ Returns a new batch """
        raw_data = np.array(self.data)
        data_len = len(raw_data)
        batch_len = data_len // self.batch_size
        data = np.reshape(raw_data[0: self.batch_size * batch_len],
                          [self.batch_size, batch_len])
        epoch_size = (batch_len - 1) // self.num_steps
        for i in xrange(epoch_size):
            x, y = data[:, i * self.num_steps:(i + 1) * self.num_steps], \
                data[:, i * self.num_steps + 1: (i + 1) * self.num_steps + 1]
            yield x, y
