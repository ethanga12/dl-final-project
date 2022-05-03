"""
This file will contain the code specific to the CNN class
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


def create_and_run_cnn(train_images, train_labels, test_images, test_labels, num_classes):
    hidden_size = 100
    model = keras.Sequential(
        layers=[
                layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(hidden_size),
                layers.Dropout(0.3),
                layers.Dense(hidden_size),
                layers.Dropout(0.3),
                layers.Dense(num_classes)
            ]
    )
    model.build(input_shape=(train_images.shape))

    model.summary()

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print("CNN TEST ACCURACY:", test_acc)


class CNNModel(tf.keras.Model):
    """An image classification model that utilizes convolutional neural networks"""

    def __init__(self):
        super(CNNModel, self).__init__()

        self.batch_size = 100
        self.num_classes = 10

        self.learning_rate = 0.05 
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.hidden_size = 100 

        self.model = tf.keras.Sequential(
            layers=[
                layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.Flatten(),
                layers.Dense(self.hidden_size),
                layers.Dropout(0.3),
                layers.Dense(self.hidden_size),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes)
            ]
        )
        

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
        return loss

    def accuracy(self, logits, labels):
        # correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        # return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return tf.reduce_sum(tf.keras.metrics.sparse_categorical_accuracy(labels, logits)) / self.batch_size
