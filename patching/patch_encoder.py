from tensorflow.keras import layers
import tensorflow as tf

class PatchEncoder(layers.Layer):
    """Makes projection of a patch and adds 
    position embedding to projected vector.

    Parameters
    ----------
    num_patches : int
        Number of patches in image.

    projection_dim : int
        Dimention of projection to make.

    Attributes
    ----------
    projection : layers.Dense
            layer for making projection of guven dimention.

    """
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        """ Calls projection & embedding to a patch.

        Parameters
        ----------
        patch : tf.Tensor
        Array of all pixels of a patch

        Attributes
        ----------
        positions : tf.range
            position of patch from 0 to (number of patches)
        encoded : float

        Returns
        -------
        encoded :  
            Embedding, which maps patch number to patch projection.
        """
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded