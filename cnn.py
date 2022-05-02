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

        self.batch_size = 64
        self.num_classes = 10

        self.learning_rate = 0.05 
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.hidden_size = 100 

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.Dense(self.hidden_size))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(self.hidden_size))
        self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.Dense(self.num_classes))


    def call(self, inputs):
        logits = self.model(inputs)
        return logits

    def loss(self, logits, labels):
        loss = tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels, logits)) / self.batch_size
        return loss

    def accuracy(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
