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
@click.option("--lstm-size", "-l", type=click.INT, help="Dimension of lstm states")
@click.option("--epochs", "-e", type=click.INT, help="No. of epochs to run training for")
def train_nn(file, batch_size, num_steps, lstm_size, epochs):
    """ Train neural network """
    ds = Dataset(file, batch_size=batch_size, num_steps=num_steps)
    n = RNN(data=ds, lstm_size=lstm_size, num_epochs=epochs)
    n.train()
