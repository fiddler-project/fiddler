class Dataset(object):

    def __init__(self, path):
        """ `path` is processed file's path """
        self.path = path

    def get_data(self, format="raw"):
        pass

    def batch(self, size=100):
        pass
