import numpy as np
import re
import io


class Dataset(object):

    def __init__(self, path, batch_size=1000, num_steps=10, with_delim=False):
        """ `path` is processed file's path """
        print("Loading dataset...")
        with io.open(path, encoding='utf-8', mode='r') as f:
            raw_data = f.read()
            print("Data length:", len(raw_data))

        self.vocab = set(raw_data)
        start_symbol, end_symbol = '<s>', '</s>'
        if with_delim:
            self.vocab.update({start_symbol, end_symbol})

        self.vocab_size = len(self.vocab)
        self.idx_to_vocab = dict(enumerate(self.vocab))
        self.vocab_to_idx = dict(
            zip(self.idx_to_vocab.values(), self.idx_to_vocab.keys()))

        if not with_delim:
            self.data = [self.vocab_to_idx[c] for c in raw_data]
        else:
            tunes = raw_data.split('\n\n')
            for t in tunes:
                self.data.append(self.vocab_to_idx[start_symbol])
                self.data.append([self.vocab_to_idx[c] for c in t])
                self.data.append(self.vocab_to_idx[end_symbol])

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
            yield x, y, epoch_size


class LMDataset(object):

    def __init__(self, file):
        self.data = {}
        with open(file, 'r') as f:
            tunes = re.findall(r'\nK:(.*)\n(.*)\n', f.read())
            for key, tune in tunes:
                if key in self.data:
                    self.data[key].append(tune)
                else:
                    self.data[key] = [tune]

    def get_data(self):
        """ Returns data as {key: list of tunes} """
        return self.data
