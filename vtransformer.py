from tabnanny import verbose
from unicodedata import mirrored
from unittest.mock import patch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
# from transformers import Transformer
import matplotlib.pyplot as plt
import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context

USE_CIFAR100 = False

# CHECK FOR GPU MODE
if (len(sys.argv) == 2) and (sys.argv[1].lower() == "gpu"):
    multigpu_mode = True
else: 
    multigpu_mode = False
num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
if multigpu_mode:
    print("MULTI-GPU MODE,", num_gpus, "GPUs FOUND")
# ==================

if USE_CIFAR100:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
else:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

# prepare the data
num_classes = 100 if USE_CIFAR100 else 10
input_shape = (32, 32, 3)
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
#hyperparameters
learning_rate = 0.001 * num_gpus if multigpu_mode else 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10
image_size = 72
patch_size = 6
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim, 
]
transformer_layers = 8
mlp_head_units = [2048, 1024]

# VISUALIZATION HELPER

def visualize_misclassified(test_images, test_labels, predictions, class_names): 
    print("Visualizing data...")
    plt.figure(figsize=(10,10))
    test_labels = tf.squeeze(test_labels)
    incorrect_pred = tf.math.subtract(predictions, test_labels)
    i = 0 
    num_plotted = 0 
    while num_plotted < 25:
        if incorrect_pred[i] != 0: 
            plt.subplot(5,5,num_plotted+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(test_images[i])
            plt.xlabel(class_names[predictions[i]])
            num_plotted += 1
        i += 1
    plt.savefig('./visualizations/transformer_misclassified.png')
    plt.show()

#data aug

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

data_augmentation.layers[0].adapt(x_train)

#mlp 

def mlp(x, hidden_units, dropout_rate):
    for unit in hidden_units:
        x = layers.Dense(unit, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x 

class Patches(layers.Layer): 
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    
    def call(self, images): 
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config

class PatchEncoder(layers.Layer): 
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim = num_patches, output_dim = projection_dim
        )
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection,
        })
        return config

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    augmented = data_augmentation(inputs)
    patches = Patches(patch_size)(augmented)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model): 
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ], 
    )

    checkpoint_filepath = "./checkpoints"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train, 
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    model.save("./saved_models/trained_keras_vt.h5")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./visualizations/vtransformer_loss.png')
    plt.show()
    pred = np.argmax(model.predict(x_test), axis = 1) 
    visualize_misclassified(x_test, y_test, pred, class_names)
    return history


def run_experiment_multi_gpu():
    # Use mirrored strategy to run on multiple GPUs
    mirrored_strategy = tf.distribute.MirroredStrategy()
    print("Number of GPUs:", mirrored_strategy.num_replicas_in_sync)

    with mirrored_strategy.scope():
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        inputs = layers.Input(shape=input_shape)
        augmented = data_augmentation(inputs)
        patches = Patches(patch_size)(augmented)
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        for _ in range(transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([attention_output, encoded_patches])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            encoded_patches = layers.Add()([x3, x2])

        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)

        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        logits = layers.Dense(num_classes)(features)
        model = keras.Model(inputs=inputs, outputs=logits)

        model.build(input_shape = x_train.shape)

        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ], 
        )
    
        checkpoint_filepath = "./checkpoints"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

    history = model.fit(
        x=x_train,
        y=y_train, 
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")
    model.save("./saved_models/trained_keras_vt.h5")

    return history


if multigpu_mode:
    history = run_experiment_multi_gpu()
else:
    vit_classifier = create_vit_classifier()
    history = run_experiment(vit_classifier)

