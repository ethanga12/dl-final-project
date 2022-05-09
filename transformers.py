"""
This file will contain the code specific to the visual transformers
class.
"""
from turtle import forward
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow import nn
from keras import layers
import numpy as np
from einops import rearrange

# About tf modules: https://www.tensorflow.org/api_docs/python/tf/Module





class BasicConvolutionBlock(tf.Module):
    """
    Layer that runs the input through convolution and dense layers.
    """
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, bias=False):
        super(BasicConvolutionBlock, self).__init__()
        self.conv1 = tf.nn.conv2d(in_planes, planes, stride, padding=1, bais=False)
        self.bn1 = tf.nn.batch_normalization(planes)
        self.conv2 = tf.nn.conv2d(planes, planes, kernel_size, stride, padding=1, bias=False)
        self.bn2 = tf.nn.batch_normalization(planes)


        self.shortcut = tf.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = keras.Sequential(
                tf.nn.conv2d(in_planes, self.expansion* planes,kernel_size, stride, bias=False),
                tf.nn.batch_normalization(self.expansion* planes)
            )
        

    def call(self, inputs):
        """forward pass for our model"""
        out = tf.nn.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(inputs)
        out = tf.nn.relu(out)

        return out

class ResidualBlock(tf.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn 
        pass

    def call(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
        

class LayerNormalize(tf.Module):
    def __init__(self, dim, fn):
        super.__init__()
        self.norm = layers.LayerNormalization()
        self.fn = fn

    def call(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(tf.Module):
    def __init__(self, dims, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = 1 / tf.square(dims) # dims ** -0.5
        self.convert_to_qkv = layers.Dense(dims * 3, use_bias=True)
        self.nn1 = layers.Dense(dims)
        self.dropout1 = layers.Dropout(dropout)

    def call(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.convert_to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = h) # split into multi-heads
        dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        # if mask is not None: WE'RE NOT DOING THIS
        #     mask = tf.pad(mask.flatten(1), (1, 0))
        attention = tf.nn.softmax(dots, axis=-1)
        out = tf.einsum('bhij,bhjd->bhid', attention, v) # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)') #concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.dropout1(out)
        return out


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
class MLP_Block(tf.Module):
    def __init__(self, dim, hidden_dim, dropout =0.1):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(hidden_dim, "gelu")
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.d2 = tf.keras.layers.Dense(dim)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
    
    def forward(self, x):
        x = self.d1(x)
        x = self.dropout(x)
        x = self.d2(x)
        x = self.dropout2(x)

        return x 

class ViTResNet(tf.Module):
    # BATCH_SIZE_TRAIN = 100
    # BATCH_SIZE_TEST = 100
    def __init__(self, block, num_blocks, num_calsses=10, dim = 128, num_tokens = 8, mlp_dim = 256, heads = 8, depth = 6, emb_dropout = 0.1, dropout = 0.1):
        super(ViTResNet, self).__init__()
        self.in_planes = 16
        self.L = num_tokens
        self.cT = dim

        self.conv1 = tf.keras.layers.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)


        #THESE ARE PYTORCH PARAMETERS SHOULD BE TRANSLATED
        self.token_wA = tf.zeros(100, self.L, 64)
        self.token_wV = tf.zeros(100, 64, self.cT)


        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides: 
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return tf.keras.layers.Sequential(*layers)

