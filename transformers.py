"""
This file will contain the code specific to the visual transformers
class.
"""
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow import nn
from keras import layers
import numpy as np



class BasicConvolutionBlock(layers.Layer):
    """
    Layer that runs the input through convolution and dense layers.
    """
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, bias=False):
        pass

    def call(self, inputs):
        pass
        

class VisualTransformerModel(keras.Model):
    """An image classification model that utilizes visual transformers"""

    def __init__(self, image_size: int, patch_size: int, num_classes: int, depth=1, heads=1):
        """
        :param image_size: greater dimension of image (height/width)
        :param patch_size: number of patches
        :param depth: number of transformer blocks
        :param heads: number of heads for attention
        """
        super(VisualTransformerModel, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Optimizer
        self.learning_rate = 0.01
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Hyperparameters
        self.hidden_size = None

        # Sequentials
        first = keras.Sequential(
            layers=[
                # filters ~= out_channels
                layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same', use_bias=False),
                layers.BatchNormalization(), # nn.BatchNorm2d(16) in original
                # Block 1 - relates to 
            ])


    def call(self, input_images):
        """
        Forward pass with the function
        :param input_images: tensor of shape [num_inputs, image_size, image_size, channels]
        :return: probabilities, shape of [num_inputs, num_classes]
        """



        pass

    def loss(self):
        pass

    def accuracy(self):
        pass

