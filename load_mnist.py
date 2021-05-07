from transformers.vision_transformer import VisionTransformer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 1
image_size = 28  # We'll resize input images to this size
patch_size = 10  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024] 

num_classes = 10
input_shape = (image_size, image_size, 1)

vit_classifier = VisionTransformer(num_classes, input_shape, patch_size, num_patches, transformer_layers, transformer_units, num_heads, mlp_head_units, projection_dim)
model = vit_classifier.create_model()
optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)


model.load_weights('pretrained/mnist/mnist.tf')

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

_, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)