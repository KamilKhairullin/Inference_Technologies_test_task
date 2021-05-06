from patching.patch_encoder import PatchEncoder
from patching.patcher import Patcher
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

class VisionTransformer():
    """Creates Vision Transformer model with given parameters

    Parameters
    ----------
    num_classes : int
        Number of classes to classify.

    input_shape : int
        Input shape of image (NxN). All images are same size.

    patch_size : int
        Size of the patch (NxN).

    projection_dim : int
        Dimention of projection to make.

    transformer_layers : int 
        Number of layers in transformer

    transformer_units : 

    num_heads : int

    mlp_head_units : [int]

    projection_dim : int
    
    """
    def __init__(self, num_classes, input_shape, patch_size, num_patches, transformer_layers, transformer_units, num_heads, mlp_head_units, projection_dim):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transformer_layers = transformer_layers
        self.transformer_units = transformer_units
        self.num_heads = num_heads
        self.mlp_head_units = mlp_head_units
        self.projection_dim = projection_dim
        
    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def create_model(self):
        inputs = layers.Input(shape=self.input_shape)
        # Create patches.
        patches = Patcher(self.patch_size)(inputs)
        # Encode patches.
        encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(self.num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model