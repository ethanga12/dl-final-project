"""
This file will contain the code specific to the CNN class
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

class CNNModel(tf.keras.Model):
    """An image classification model that utilizes convolutional neural networks"""

    def __init__(self):
        super(CNNModel, self).__init__()

        self.batch_size = 100
        self.num_classes = 10

        self.learning_rate = 0.05 
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.hidden_size = 100 

        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(self.hidden_size))
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Dense(self.hidden_size))
        self.model.add(layers.Dropout(0.1))
        self.model.add(layers.Dense(self.num_classes))

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    def call(self, inputs):
        logits = self.model(inputs)
        return logits

    def loss(self, logits, labels):
        # loss = tf.math.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels, logits)) #/ 100 #change back to batch size from 100
        loss = tf.keras.metrics.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return tf.reduce_sum(loss)

    def accuracy(self, logits, labels):
        # correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        # return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(labels, logits))
