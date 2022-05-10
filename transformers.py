"""
This file will contain the code specific to the visual transformers
class.
"""
from turtle import forward
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow import nn
from tensorflow.keras import layers
import numpy as np
import time
from einops import rearrange

# About tf modules: https://www.tensorflow.org/api_docs/python/tf/Module

class BasicBlock(tf.Module):
    """
    Layer that runs the input through convolution and dense layers.
    """
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, bias=False):
        super(BasicBlock, self).__init__()
        # self.conv1 = layers.Conv2D(in_planes, planes, strides=stride, padding="same", use_bias=False)
        self.conv1 = layers.Conv2D(planes, kernel_size, strides=stride, padding="same", use_bias=False)
        # self.bn1 = layers.BatchNormalization(planes)
        self.conv2 = layers.Conv2D(planes, kernel_size, strides=stride, padding="same", use_bias=False)
        # self.conv2 = layers.Conv2D(planes, planes, kernel_size, strides=stride, padding="same", use_bias=False)
        # self.bn2 = layers.BatchNormalization(planes)

        self.shortcut = keras.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = keras.Sequential(
                layers=[
                    # layers.Conv2D(in_planes, self.expansion * planes, kernel_size, stride, use_bias=False),
                    layers.Conv2D(self.expansion * planes, kernel_size, stride, padding="same", use_bias=False),
                    layers.BatchNormalization(self.expansion * planes)
                    # layers.BatchNormalization tf.nn.batch_normalization(self.expansion* planes)
                    # tf.nn.batch_normalization(self.expansion* planes)
                ])
        

    def call(self, inputs):
        """forward pass for our model"""
        out = tf.nn.relu(self.conv1(inputs))
        mean, var = tf.nn.moments(out, [0, 1, 2])
        out = tf.nn.batch_normalization(out, mean, var, 0, 1, variance_epsilon=1e-5)
        out = self.conv2(out)
        out = tf.nn.batch_normalization(out, mean, var, 0, 1, variance_epsilon=1e-5)
        # out += self.shortcut(inputs) MORE BATCH NORMALIZATION STUFF
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
        self.norm = layers.LayerNormalization(dim)
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

class Transformer(keras.Model):
    """An image classification model that utilizes visual transformers"""

    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        """
        :param dim: greater dimension of image (height/width)
        :param depth: number of transformer blocks
        :param heads: number of heads for attention
        :param mlp_dim: dimensions for mlp block
        :param dropout: dropout amount
        """
        super().__init__()
        self.transformer_layers = []
        for _ in range(depth):
            self.transformer_layers.append(Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))))
            self.transformer_layers.append(Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout))))

    def call(self, x, mask=None):
        """
        Forward pass with the function
        :param x: tensor of shape [num_inputs, image_size, image_size, channels]
        """
        for attention, mlp in self.transformer_layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x

class Residual(tf.Module):
    def __init__(self, fn):
        super().__init__()
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
    def __init__(self, block, num_blocks, num_classes=10, dim = 128, num_tokens = 8, mlp_dim = 256, heads = 8, depth = 6, emb_dropout = 0.1, dropout = 0.1):
        super(ViTResNet, self).__init__()

        self.batch_size = 100
        self.in_planes = 16
        self.L = num_tokens
        self.cT = dim

        # self.conv1 = tf.keras.layers.Conv2D(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # Is this supposed to be valid or same padding
        self.conv1 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding="same", use_bias=False)
        # self.bn1 = tf.nn.batch_normalization()
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        #THESE ARE PYTORCH PARAMETERS SHOULD BE 
        self.token_wA = tf.Variable(tf.zeros((100, self.L, 64)))
        self.token_wV = tf.Variable(tf.zeros((100, 64, self.cT)))

        self.pos_embedding = tf.Variable(tf.random.normal((1, (num_tokens + 1), dim), stddev =.02)) #MIGHT WANT THIS TO BE RANDOM DISTRIBUTION
        
        self.cls_token = tf.Variable(tf.zeros(1, 1, dim))
        self.dropout = layers.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = lambda x: tf.identity(x)
        # tf.identity() #OF WHAT THOOOOO
        
        self.d1 = layers.Dense(num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides: 
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        # return keras.Sequential(*layers)
        return keras.Sequential(layers)

    def call(self, img, mask=None):
        x = self.conv1(img)
        mean, var = tf.nn.moments(x, [0, 1, 2])
        x = tf.nn.batch_normalization(x, mean, var, 0, 1, variance_epsilon=1e-5)
        x = tf.nn.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = rearrange(x, 'b c h w -> b (h w) c') # 64 vectors each with 64 points. These are the sequences or word vectors like in NLP
        wa = rearrange(self.token_wA, 'b h w -> b w h')
        A= tf.einsum('bij,bjk->bik', x, wa) 
        A = rearrange(A, 'b h w -> b w h') # Transpose
        A = A.softmax(axis=-1)

        VV= tf.einsum('bij,bjk->bik', x, self.token_wV)       
        T = tf.einsum('bij,bjk->bik', A, VV)  

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = tf.concat((cls_tokens, T), axis=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x, mask) # main game
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

def train(model, opt, train_inputs, train_labels, loss_history):
    num_images = train_inputs.shape[0]

    for i in range (model.batch_size, num_images, model.batch_size):
        batch_inputs = train_inputs[i - model.batch_size: i, :, :, :]
        batch_labels = train_labels[i - model.batch_size: i, :]
        with tf.GradientTape() as tape:
            logits = tf.nn.log_softmax(model.call(batch_inputs), axis=1)
            # logits = tf.nn.log_softmax(model.call(batch_inputs))
            loss = tf.experimental.nn.losses.negloglik(logits, batch_labels)

        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

        if i % 200 == 0:
            print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (i, float(loss))
                )
            print("Seen so far: %s samples" % ((i + 1) * model.batch_size))

# def train(model, opt, data_loader, loss_history):
#     num_samples = len(data_loader.dataset)
#     model.train()

#     for i, (data, target) in enumerate(data_loader):
#         with tf.GradientTape as tape:
#             logits = tf.nn.log_softmax(model(data), axis=1)
#             loss = tf.experimental.nn.losses.negloglik(logits, target)
    
#         grads = tape.gradient(loss, model.trainable_weights)
#         opt.apply_gradients(zip(grads, model.trainable_weights))

#         if i % 200 == 0:
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                 % (i, float(loss))
#             )
#             print("Seen so far: %s samples" % ((i + 1) * model.batch_size))
            
def evaluate(model, test_inputs, test_labels, loss_history): 
    # model.eval() - in PyTorch this notifies the model that we're in eval mode - not sure how to do this in tf

    num_images = test_labels.shape[0]
    correct_samples = 0 
    total_loss = 0

    for i in range (model.batch_size, num_images, model.batch_size):
        batch_inputs = test_inputs[i - model.batch_size: i, :, :, :]
        batch_labels = test_labels[i - model.batch_size: i, :]
        output = tf.nn.log_softmax(model.call(batch_inputs), axis=1) # need to specify axis=1? 
        total_loss += tf.keras.metrics.sparse_categorical_crossentropy(batch_labels, output, from_logits=True)
        pred = tf.math.reduce_max(output, axis=1) # need to say axis=1? 
        correct_samples += tf.math.reduce_sum(tf.math.equal(pred, batch_labels))
    
    avg_loss = total_loss / num_images
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(num_images) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / num_images) + '%)\n')


# Main


def create_and_run_vtmodel(train_images, train_labels, test_images, test_labels):
    train_loss_history, test_loss_history = [], []
    model = ViTResNet(BasicBlock, [3, 3, 3])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        print("Current Epoch:", epoch)
        start_time = time.time()
        train(model, optimizer, train_images, train_labels, train_loss_history)
        print(f"Epoch", epoch, "finished in", '{:5.2f}'.format(time.time() - start_time), "seconds")
        evaluate(model, test_images, test_labels, test_loss_history)