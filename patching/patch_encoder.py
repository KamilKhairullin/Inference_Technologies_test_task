from tensorflow.keras import layers
import tensorflow as tf

class PatchEncoder(layers.Layer):
    """The PatchEncoder layer will linearly transform a patch by 
    projecting it into a vector of size projection_dim. In addition,
    it adds a learnable position embedding to the projected vector.

    Parameters
    ----------
    num_patches : int
        Number of patches in image.

    projection_dim : int
        Dimention of projection to make.

    Attributes
    ----------
    projection : layers.Dense
            Layer through which we pass data and get projection of given dimention.
    
    position_embedding : layers.Embedding
            From array with positions of patches produces embedding of given dimention.
            Example: (2d embedding)
            [5, 1, 2, 3, 6] -> [[0.7, 1.7], [0.1, 4.2], [1.0, 3.1], [0.3, 2.1], [4.1, 2.0]]
            And theese encoding attached to image patches.

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

        Attributes
        ----------
        positions : tf.range
            position of patch from 0 to (number of patches).

        Returns
        -------
        encoded :  
            Embedding, which maps patch number to patch projection.
        """
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        """ To be able to save our model, we need to override this method in custom layers like this.

        Returns
        -------
        config with added __init__ attributes of this class.
        """
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection' : self.projection,
            'position_embedding' : self.position_embedding
        })
        return config