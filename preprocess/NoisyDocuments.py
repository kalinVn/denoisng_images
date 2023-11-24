import os

from service.Drive import Drive

import numpy as np
from config import CLEAN_DOCUMENTS_IMG_PATH, NOISY_DOCUMENTS_IMG_PATH

class NoisyDocuments:

    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.x_train_reshaped = None
        self.x_test_reshaped = None
        self.x_train_scaled = None
        self.x_test_scaled = None

        self.x_train_noisy = []
        self.x_test_noisy = []
        self.x_train_clean = []
        self.x_test_clean = []
        self.drive = Drive()

    def train_test_split(self):
        target_size = (420, 540)
        path = NOISY_DOCUMENTS_IMG_PATH
        self.x_train_noisy = self.drive.create_img_array_by_folder(path, target_size)
        path = CLEAN_DOCUMENTS_IMG_PATH
        self.x_train_clean = self.drive.create_img_array_by_folder(path, target_size)

        return self.x_train_noisy, self.x_train_clean, [], []

    def set_noisy_images(self):
        self.x_train_noisy = np.array(self.x_train_noisy)
        self.x_train_clean = np.array(self.x_train_clean)

        self.x_test_noisy = self.x_train_noisy[0:20]
        self.x_train_noisy = self.x_train_noisy[21:50]

        self.x_test_clean = self.x_train_clean[0:20]
        self.x_train_clean = self.x_train_clean[21:50]

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

    def get_x_train_clean(self):
        return self.x_train_clean

    def get_x_test(self):
        return self.x_test

    def get_y_test(self):
        return self.y_test

    def get_x_train_noisy(self):
        return self.x_train_noisy

    def get_x_test_noisy(self):
        return self.x_test_noisy

    def get_x_test_clean(self):
        return self.x_test_clean

