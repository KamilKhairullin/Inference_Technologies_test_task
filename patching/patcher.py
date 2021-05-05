class Patcher(layers.Layer):
    """Split image into patches.

    Parameters
    ----------
    image_size : int
        Size of the image (NxN).

    patch_size : int
        Size of the patch (NxN).

    channels : int
        Number of channels in image (1 for gray, 3 for RGB etc.).
    
    Attributes
    ----------
    num_patches : int
        Number of patches
    """
    def __init__(self, patch_size, channels = 1):
        super(Patcher, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.channels = channels
  
    def call(self, images):
        """Calls image patching 

        Parameters
        ----------
        images : tf.image
            Images to be patched.

        Attributes
        ----------
        sizes : [1, size_rows, size_cols, 1]
            The size of the extracted patches.

        strides : [1, stride_rows, stride_cols, 1]
            How far the centers of two consecutive patches are in the images.
            Make patches overlapped, if not equal to patch size. 

        rates : [1, rate_rows, rate_cols, 1]
            How far two consecutive patch samples are in the input.
            [1, 1, 1, 1] is defalut.

        padding : String
            Padding algorithm to use.

        batch_size : int
            Number of images in batch.
        
        patch_dims : int
            (Number of pixels in patch) * (number of channels)

        Returns
        -------
        patches : tf.Tensor
            A 4-D Tensor
        """
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self.patch_size, self.patch_size, 1],
            strides = [1, self.patch_size, self.patch_size, 1],
            rates = [1, 1, 1, 1],
            padding = "VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches