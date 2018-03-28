class Dataset(object):

    def __init__(self, path, batch_size=1000):
        """ `path` is processed file's path """
        self.path = path
        self.batch_size = batch_size

    def get_data(self, format="raw"):
        pass

    def batch(self):
        """ Returns a new batch """
        pass
