"""
Entry point for training models and generating music sheet!
"""

import click
import numpy as np
from network import RNN
from dataset import Dataset
import tensorflow as tf


@click.group()
def main():
    pass


@main.command()
@click.option("--file", "-f", type=click.Path(exists=True), help="Train Data File Path")
@click.option("--batch-size", "-b", type=click.INT, help="Batch size")
@click.option("--num-steps", "-n", type=click.INT, help="No. of time steps in RNN")
@click.option("--cell-size", "-s", type=click.INT, help="Dimension of cell states")
@click.option("--epochs", "-e", type=click.INT,
              help="No. of epochs to run training for")
@click.option("--cell", "-c", type=click.Choice(['lstm', 'gru']),
              default="lstm", help="Type of cell used in RNN")
@click.option("--test-seed", "-t", help="Seed input for printing predicted text after each training step")
def train_rnn(file, batch_size, num_steps, cell_size, epochs, cell, test_seed):
    """ Train neural network """
    ds = Dataset(file, batch_size=batch_size, num_steps=num_steps)
    n = RNN(data=ds, cell=cell, cell_size=cell_size, num_epochs=epochs)
    n.train(test_output=True, test_seed=test_seed)