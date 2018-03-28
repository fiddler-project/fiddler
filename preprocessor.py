""" Preprocess the dataset and store it on file in the vectorized format """
from abc import ABCMeta, abstractmethod


class Preprocessor(object):
    """ Parent preprocessor abstract class """
    __meta__ = ABCMeta

    def __init__(self, data_path):
        """ `data_path` is path to the json file containing raw data """
        self.data_path = data_path

    @abstractmethod
    def process(self, store=True):
        raise NotImplementedError("Method not implemented yet!")


class NNPreprocessor(Preprocessor):
    """ concrete preprocessor for neural network based methods """
    pass


class LMPreprocessor(Preprocessor):
    """ concrete preprocessor for language model """
    pass
