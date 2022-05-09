"""
This file will contain all the code used to train either the CNN or
transformers model as well as any code to visualize results.
"""

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import ssl
import tqdm # for progress bar
import sys
import time

from transformers import VisualTransformerModel
from cnn import *

# The following line is to allow us to load the cifar10 dataset without errors
ssl._create_default_https_context = ssl._create_unverified_context

# TRAIN

def train(model: keras.Model, train_inputs, train_labels, batch_size):
    num_entries = train_labels.shape[0]

    shuffle = tf.random.shuffle(np.arange(num_entries))
    shuffled_inputs = tf.gather(train_inputs, shuffle)
    shuffled_labels = tf.gather(train_labels, shuffle)

    for i in range (batch_size, num_entries, batch_size):
        batch_inputs = shuffled_inputs[i - batch_size: i, :, :, :]
        batch_labels = shuffled_labels[i - batch_size: i, :]
        with tf.GradientTape() as tape:
            predictions = model.call(batch_inputs) 
            loss = model.loss(predictions, batch_labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#################

# TEST

def test(model: keras.Model, test_inputs, test_labels, batch_size):
    num_entries = test_labels.shape[0]
    acc = 0

    for i in range (batch_size, num_entries, batch_size):
        batch_inputs = test_inputs[i - batch_size: i, :, :, :]
        batch_labels = test_labels[i - batch_size: i, :]
        pred = model.call(batch_inputs)
        acc += model.accuracy(pred, batch_labels) 
    
    return acc / (num_entries // batch_size)

#################

# VISUALIZER FUNCTIONS

def visualize_inputs(class_names, images, labels):
    print("Visualizing data...")
    plt.figure(figsize=(10,10))
    labels = tf.squeeze(labels)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]])
    plt.show()

#################

# LOAD DATA

def load_cifar_data():
    """
    Loads the data from the CIFAR10 dataset.
    Normalizes pixel values to be in between 0 and 1.
    :return: class_names (list), train_images, train_labels, test_images, test_labels <= tensors
    """
    print("Loading CIFAR10 Dataset...")
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    print("CIFAR10 Dataset loaded...")
    return class_names, train_images, train_labels, test_images, test_labels

#################

# MAIN

def main():
    class_names, train_images, train_labels, test_images, test_labels = load_cifar_data()
    # Determine whether to train a new model or use the saved one
    if len(sys.argv) == 2 and sys.argv[0].lower() == "traincnn":
        create_and_run_cnn(train_images, train_labels, test_images, test_labels, num_classes=10)
    else:
        try:
            # If no options given, try loading the model first
            print("Attempting to load saved model:")
            load_cnn(test_images, test_labels)
        except:
            print("Failed to load saved model, training new model")
            time.sleep(1)
            create_and_run_cnn(train_images, train_labels, test_images, test_labels, num_classes=10)

    print("Execution successful! Exiting...")
    
    return

    cnn = CNNModel()
    
    num_epochs = 10

    cnn.model.build(input_shape=(train_images.shape))

    cnn.model.summary()

    cnn.model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = cnn.fit(train_images, train_labels, epochs=num_epochs,
    validation_data=(test_images, test_labels))

    test_loss, test_acc = cnn.evaluate(test_images,  test_labels, verbose=2)


    # for i in range(num_epochs): 
    #     indices = tf.random.shuffle(tf.Variable(np.arange(train_images.shape[0]))) 
    #     train(model, tf.gather(train_images, indices), tf.gather(train_labels, indices), model.batch_size)

    # accuracy = test(model, test_images, test_labels, model.batch_size)
    # print("CNN Accuracy: ", accuracy)

#################

if __name__ == '__main__':
    main()
