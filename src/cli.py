"""
Entry point for training models and generating music sheet!
"""

import click
import numpy as np
from network import RNN
from languagemodel import NgramLM
from dataset import Dataset, LMDataset
import pickle


@click.group()
def main():
    pass


@main.command()
@click.option("--file", "-f", type=click.Path(exists=True),
              help="Train Data File Path", required=True)
@click.option("--batch-size", "-b", default=50, type=click.INT, help="Batch size")
@click.option("--layers", "-l", default=2, type=click.INT, help="Number of layers in the network")
@click.option("--learning-rate", "-r", default=1e-3, type=click.FLOAT, help="Learning Rate")
@click.option("--num-steps", "-n", type=click.INT, default=15, help="No. of time steps in RNN")
@click.option("--cell-size", "-s", type=click.INT, default=100, help="Dimension of cell states")
@click.option("--dropout", "-d", type=click.FLOAT, default=0.8, help="Dropout probability for the output")
@click.option("--epochs", "-e", type=click.INT, default=10,
              help="No. of epochs to run training for")
@click.option("--cell", "-c", type=click.Choice(['lstm', 'gru']),
              default="lstm", help="Type of cell used in RNN")
@click.option("--test-seed", "-t", help="Seed input for printing predicted text after each training step")
@click.option("--delim/--no-delim", default=True, help="Delimit tunes with start and end symbol")
@click.option("--save/--no-save", default=True, help="Save model to file")
def train_rnn(file, batch_size, layers, learning_rate, dropout,
              num_steps, cell_size, epochs, cell, test_seed, delim, save):
    """ Train neural network """
    model_name = "cell-{}-size-{}-batch-{}-steps-{}-layers-{}-lr-{}-dropout-{}".format(
        cell, cell_size, batch_size, num_steps, layers, learning_rate, dropout)
    ds = Dataset(file, batch_size=batch_size,
                 num_steps=num_steps, with_delim=delim)
    n = RNN(data=ds, cell=cell, num_layers=layers, dropout=dropout,
            learning_rate=learning_rate, cell_size=cell_size, num_epochs=epochs)
    n.train(save=save, model_name=model_name, test_output=True,
            test_seed=test_seed, with_delim=delim)
    if save:
        n.save(model_name)


@main.command()
@click.option("--model_path", "-m", type=click.Path(exists=True),
              help="Directory path for saved model")
def generate(model_path):
    with open('{}/rnn.pickle'.format(model_path)) as f:
        config = pickle.load(f)
    n = RNN(training=False, **config)
    print n.gen_text(sess=None, model_path=model_path)


@main.command()
@click.option("-n", type=click.INT, 
              help="N for N-gram", required=True)
@click.option("--file", "-f", type=click.Path(exists=True),
              help="Train Data File Path", required=True)
@click.option("--gen/--no-gen", "-g", default=False, 
              help="True if tune needs to be generated")
@click.option("--test/--no-test", default=False, 
              help="True if test needs to be performed")
@click.option("--smoothing", "-s", type=click.Choice(['add-k']),
              help="Select smoothing")
@click.option("-k", type=click.FLOAT, help="k for add-k smoothing")
def ngram_lm(file, n, gen, test, smoothing, k):
  lmd = LMDataset(file)
  split = int(len(lmd.raw_data)*0.8)
  train, test = lmd.raw_data[:split], lmd.raw_data[split:]
  lm = NgramLM(data=train, n=n)
  lm.train()
  if gen:
    print 'Generated Tune: \n{}'.format(lm.generate())
  if test:
    acc, pp = lm.test(data=test, s=smoothing, k=k)
    print 'Accuracy: {}\nPerplexity: {}'.format(acc, pp)

if __name__ == '__main__':
    main()