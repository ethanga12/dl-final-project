"""
This file will contain the code specific to the CNN class
"""
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

class CNNModel(keras.Model):
    """An image classification model that utilizes convolutional neural networks"""

    def init(self):
        super(CNNModel, self).__init__()

    def call(self, inputs):
        pass

    def loss(self):
        pass

    def accuracy(self):
        pass
