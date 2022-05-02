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

# About tf modules: https://www.tensorflow.org/api_docs/python/tf/Module





class BasicConvolutionBlock(tf.Module):
    """
    Layer that runs the input through convolution and dense layers.
    """
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, bias=False):
        pass

    def call(self, inputs):
        pass

class ResidualBlock(tf.Module):
    def __init__(self):
        pass

    def call(self):
        pass

class LayerNormalize(tf.Module):
    pass

class MLP_Block(tf.Module):
    pass


class Attention(tf.Module):
    def __init__(self, dims, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = 1 / tf.square(dims) # dims ** -0.5
        # TODO

class Transformer(tf.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int, dropout):
        super().__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append(ResidualBlock(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout)))),
            self.layers.append(ResidualBlock(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout))))

    def call(self, x, mask=None):
        """Apply attention layer and MLP layer"""
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x

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

class Residual(tf.Module):
    def __init__(self, fn):
        super().__init()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(tf.Module): 
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(dim)
        self.fn = fn
    def forward(self, x, **kwargs): 
        return self.fn(self.norm(x), **kwargs)

# TODO: ETHAN FINISH MLP_BLOCK
# class MLP_Block(tf.Module):
#     def __init__(self, dim, hidden_dim, dropout =0.1):
#         super().__init__()
#         self.d1 = tf.keras.
