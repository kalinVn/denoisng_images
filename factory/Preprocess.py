from builder import SequentialModel
from preprocess import MnistDigits
from preprocess import NoisyDocuments

class Preprocess:

    @staticmethod
    def get_preprocess(name):

        if name == 'mnist_digits':
            return MnistDigits()
        elif name == 'noisy_documents':
            return NoisyDocuments()

        return None


