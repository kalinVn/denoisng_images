import os.path

import tensorflow as tf

# from .Preprocess import Preprocess
from .Drive import Drive
from factory import Model as ModelFactory, Preprocess as PreprocessFactory

import numpy as np

class NN:

    def __init__(self, preprocess_name, model_name='denoise_autoencoder', hidden_layer_size=32):
        self.conv_model_2D_1 = None
        self.conv_model_2D_2 = None
        self.model = None
        self.x_test = None
        self.x_train = None
        self.y_train = None
        self.y_test = None
        self.preprocessFactory = PreprocessFactory()
        self.model_factory = ModelFactory()
        self.preprocess = self.preprocessFactory.get_preprocess(preprocess_name)
        self.output = None
        self.hidden_layer_size = hidden_layer_size
        self.model_name = model_name

    def set_train_test_data(self):
        self.x_train, self.y_train, self.x_test, self.y_test = self.preprocess.train_test_split()

    def fit(self):
        params = {
            'model_name': 'base_denoise_autoencoder',
            'hidden_layer_size': 32
        }

        self.model = self.model_factory.get_keras_model(params)

        if (os.path.exists('store/models/saved_train_model_noise_images')):
            self.model = tf.keras.models.load_model("store/models/saved_train_model_noise_images")
        else:
            self.model.fit(self.x_train, self.x_train, epochs=10)
            self.model.save('store/models/saved_train_model_noise_images')

    def predict(self):
        self.output = self.model.predict(self.x_train)

    def get_output(self):
        return self.output

    def get_x_train(self):
        return self.x_train

    def get_y_train(self):
        return self.y_train

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test

    def get_preprocess(self):
        return self.preprocess

