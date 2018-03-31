import numpy as np
from network import RNN


def batch_producer(raw_data, batch_size, num_steps):
    # raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    raw_data = np.array(raw_data)
    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])
    epoch_size = (batch_len - 1) // num_steps
    for i in xrange(epoch_size):
        x, y = data[:, i * num_steps:(i + 1) * num_steps], \
            data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
        yield x, y


if __name__ == '__main__':
    file_name = 'tinyshakespeare.txt'

    with open(file_name, 'r') as f:
        raw_data = f.read()
        print("Data length:", len(raw_data))

    vocab = set(raw_data)
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))

    data = [vocab_to_idx[c] for c in raw_data]
    del raw_data
    BATCH_SIZE = 32
    NUM_STEPS = 10
    n = RNN(data=data, producer=batch_producer, 
                batch_size=BATCH_SIZE, lstm_size=200, num_steps=NUM_STEPS,
                num_classes=vocab_size, num_epochs=1)
    n.train()
