from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense

class SequentialModel:

    @staticmethod
    def get_layer_by_type(params):

        if params['type'] == 'dense':
            args = params['args']
            return Dense(**args)
        elif params['type'] == 'conv2D':
            args = params['args']

            return Conv2D(**args)

        return None

    @staticmethod
    def build(params):
        model = Sequential()

        for layer in params['layers']:
            current_layer = SequentialModel.get_layer_by_type(layer)
            model.add(current_layer)

        model.compile(optimizer=params['optimizer'], loss=params['loss'])

        if model is not None:
            return model


        return None
