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

# About tf modules: https://www.tensorflow.org/api_docs/python/tf/Module

class BasicBlock(tf.Module):
    """
    Layer that runs the input through convolution and dense layers.
    """
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1, bias=False):
        super(BasicBlock, self).__init__()
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
        self.layers = []
        for _ in range(depth):
            self.layers.append(Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))))
            self.layers.append(Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout))))

    def call(self, x, mask=None):
        """
        Forward pass with the function
        :param x: tensor of shape [num_inputs, image_size, image_size, channels]
        """
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)
            x = mlp(x)
        return x

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
        self.token_wA = tf.Variable(tf.zeros(100, self.L, 64))
        self.token_wV = tf.Variable(tf.zeros(100, 64, self.cT))

        self.pos_embedding = tf.Variable(tf.random.normal((1, (num_tokens + 1), dim)), stddev =.02) #MIGHT WANT THIS TO BE RANDOM DISTRIBUTION
        
        self.cls_token = tf.Variable(tf.zeros(1, 1, dim))
        self.dropout = tf.keras.layers.dropout(emb_dropout)

        

        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides: 
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return tf.keras.layers.Sequential(*layers)
<<<<<<< HEAD

=======
>>>>>>> c23105c6ba25cd8e8cfd16592a06b8b9d9f74a87

def train(model, opt, data_loader, loss_history):
    num_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        with tf.GradientTape as tape:
            logits = tf.nn.log_softmax(model(data), axis=1)
            loss = tf.experimental.nn.losses.negloglik(logits, target)
    
        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

        if i % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (i, float(loss))
            )
            print("Seen so far: %s samples" % ((i + 1) * model.batch_size))
            
def evaluate(model, test_inputs, test_labels, loss_history): 
    # model.eval() - in PyTorch this notifies the model that we're in eval mode - not sure how to do this in tf

    num_images = test_labels.shape[0]
    correct_samples = 0 
    total_loss = 0 

    for i in range (model.batch_size, num_images, model.batch_size):
        batch_inputs = test_inputs[i - model.batch_size: i, :, :, :]
        batch_labels = test_labels[i - model.batch_size: i, :]
        output = tf.nn.log_softmax(model.call(batch_inputs)) # need to specify axis=1? 
        total_loss += tf.keras.metrics.sparse_categorical_crossentropy(batch_labels, output, from_logits=True)
        pred = tf.math.reduce_max(output) # need to say axis=1? 
        correct_samples += tf.math.reduce_sum(tf.math.equal(pred, batch_labels))
    
    avg_loss = total_loss / num_images
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(num_images) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / num_images) + '%)\n')


# Main

def create_and_run_vtmodel(train_images, train_labels, test_images, test_labels, num_classes=10):
    model = ViTResNet(BasicBlock, [3, 3, 3])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    num_epochs = 1
    for epoch in range(1, num_epochs + 1):
        print("Current Epoch: ", epoch)