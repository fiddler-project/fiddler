import numpy as np
from network import RNN
from dataset import Dataset
import tensorflow as tf


if __name__ == '__main__':
    file_name = 'ttttshakes.txt'    
    
    BATCH_SIZE = 200
    NUM_STEPS = 10
    ds = Dataset(file_name, batch_size=BATCH_SIZE, num_steps=NUM_STEPS)
    n = RNN(data=ds, lstm_size=300, num_epochs=2)
    n.train()