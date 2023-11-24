from keras.datasets import mnist
import numpy as np


class MnistDigits:

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.x_train_reshaped = None
        self.x_test_reshaped = None
        self.x_train_scaled = None
        self.x_test_scaled = None
        self.x_train_clean = None
        self.x_train_noisy = None
        self.x_test_noisy = None

    def train_test_split(self):
        training_set, testng_set = mnist.load_data()
        self.x_train, self.y_train = training_set
        self.x_test, self.y_test = testng_set

        self.x_train_reshaped = self.x_train.reshape((self.x_train.shape[0], self.x_train.shape[1] * self.x_train.shape[2]))
        self.x_test_reshaped = self.x_test.reshape((self.x_test.shape[0], self.x_test.shape[1] * self.x_test.shape[2]))

        self.x_train_scaled = self.x_train_reshaped / 255
        self.x_test_scaled = self.x_test_reshaped / 255

        return self.x_train_scaled, self.y_train, self.x_test_scaled, self.y_test

    def set_noisy_images(self):
        self.x_train_noisy = self.x_train_scaled + np.random.normal(0, 0.5, size=self.x_train_scaled.shape)
        self.x_test_noisy = self.x_test_scaled + np.random.normal(0, 0.5, size=self.x_test_scaled.shape)

        self.clip()

    def reshape_train_data(self, args):
        self.x_train_noisy = self.x_train_noisy.reshape(*args)
        self.x_train_scaled = self.x_train_scaled.reshape(*args)

    def clip(self):
        self.x_train_noisy = np.clip(self.x_train_noisy, a_min=0, a_max=1)
        self.x_test_noisy = np.clip(self.x_test_noisy, a_min=0, a_max=1)

    def get_x_train(self):
        return self.x_train

    def get_x_train_reshaped(self):
        return self.x_train_reshaped

    def get_x_train_scaled(self):
        return self.x_train_scaled

    def get_y_train(self):
        return self.y_train

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test

    def get_x_train_noisy(self):
        return self.x_train_noisy

    def get_x_test_noisy(self):
        return self.x_test_noisy

    def get_x_train_clean(self):
        return self.x_train_scaled


