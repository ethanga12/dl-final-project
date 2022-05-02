"""
This file will contain all the code used to train either the CNN or
transformers model as well as any code to visualize results.
"""

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import ssl
import tqdm # for progress bar

from transformers import VisualTransformerModel
from cnn import CNNModel

# The following line is to allow us to load the cifar10 dataset without errors
ssl._create_default_https_context = ssl._create_unverified_context

def train(model: keras.Model, train_inputs, train_labels):
    pass

def test(model: keras.Model, test_inputs, test_labels):
    pass

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


# VISUALIZER FUNCTIONS

def visualize_inputs(class_names, images, labels):
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


def main():
    class_names, train_images, train_labels, test_images, test_labels = load_cifar_data()
    visualize_inputs(class_names, test_images, test_labels)

if __name__ == '__main__':
    main()
