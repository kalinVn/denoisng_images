import os.path

import tensorflow as tf
from service import NN

from config import PRETRAINED_MODELS_DIR_PATH

class NN_Denoise_Autoencoder(NN):

    def __init__(self,
                 model_name='base_denoise_autoencoder',
                 preprocess_name='noisy_documents',
                 saved_model_name='conv_saved_train_model_noise_documents',
                 optimizer='Adam',
                 loss='binary_crossentropy',
                 features_dimension=(60000, 28, 28, 1),
                 epochs=10,
                 layers=[]
                 ):
        self.model_name = model_name
        self.preprocess_name = preprocess_name
        self.saved_model_name = saved_model_name
        self.layers = layers
        self.optimizer = optimizer
        self.features_dimension = features_dimension
        self.epochs = epochs
        self.loss = loss
        self.pretrained_model_dir_path = PRETRAINED_MODELS_DIR_PATH + self.saved_model_name
        NN.__init__(self, preprocess_name, self.model_name)

        NN.__init__(self, preprocess_name, self.model_name)

    def set_train_test_data(self):
        NN.set_train_test_data(self)

        self.preprocess.set_noisy_images()

    def fit(self):
        params = {
            'model_name': self.model_name,
            'layers': self.layers,
            'optimizer': self.optimizer,
            'loss': self.loss
        }

        self.model = self.model_factory.get_keras_model(params)

        x_train_noisy = self.preprocess.get_x_train_noisy()

        if (os.path.exists(self.pretrained_model_dir_path)):
            self.model = tf.keras.models.load_model(self.pretrained_model_dir_path)
        else:
            self.model.fit(x_train_noisy, self.x_train, epochs=self.epochs)
            self.model.save(self.pretrained_model_dir_path)
