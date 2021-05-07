from transformers.vision_transformer import VisionTransformer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

def run_experiment(model):
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

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        trainFlow,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data = testFlow,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


# Model / data parameters

# the data, split between train and test sets
def mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

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

trainGenerator = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=2, 
    zoom_range = 0.2,  
    vertical_flip=True,
    horizontal_flip=True
)

trainPath = '/Users/kamil/Downloads/fruits-360/Training'
testPath = '/Users/kamil/Downloads/fruits-360/Test'

testGenerator = ImageDataGenerator(rescale=1. / 255)

trainFlow = trainGenerator.flow_from_directory(
    trainPath,
    target_size=(image_size, image_size),batch_size=batch_size)

testFlow = testGenerator.flow_from_directory(
    testPath,
    target_size=(image_size, image_size),batch_size=batch_size)



vit_classifier = VisionTransformer(num_classes, input_shape, patch_size, num_patches, transformer_layers, transformer_units, num_heads, mlp_head_units, projection_dim)
model = vit_classifier.create_model()
history = run_experiment(model)