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
image_size = 100  # We'll resize input images to this size
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
num_classes = 131
input_shape = (image_size, image_size, 3)

vit_classifier = VisionTransformer(num_classes, input_shape, patch_size, num_patches, transformer_layers, transformer_units, num_heads, mlp_head_units, projection_dim)
model = vit_classifier.create_model()
optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)

model.compile(
    optimizer=optimizer,
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy"),
    ],
)


model.load_weights('pretrained/fruits/model.tf')


test_path = 'fruits-360/Test'
testGenerator = ImageDataGenerator(rescale=1. / 255)

testFlow = testGenerator.flow_from_directory(
    test_path,
    target_size=(image_size, image_size),batch_size=batch_size)

_, accuracy, top_5_accuracy = model.evaluate(testFlow)