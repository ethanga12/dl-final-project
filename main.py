"""
This file will contain all the code used to train either the CNN or
transformers model as well as any code to visualize results.
"""

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

from transformers import VisualTransformerModel
from cnn import CNNModel


def train(model: keras.Model, train_inputs, train_labels):
    pass

def test(model: keras.Model, test_inputs, test_labels):
    pass

def preprocess():
    pass

def main():
    print("hello world")

if __name__ == '__main__':
    main()
