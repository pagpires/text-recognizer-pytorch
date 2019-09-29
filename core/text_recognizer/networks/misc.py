import torch

def slide_window(images, window_width, window_stride):
    """
    Input: (batch, c, h, w)
    Output (batch, patches, c, h, w)
    'valid' drops the right-most cols, no padding
    """
    

    # """
    # Takes (image_height, image_width, 1) input,
    # Returns (num_windows, image_height, window_width, 1) output, where
    # num_windows is floor((image_width - window_width) / window_stride) + 1
    # """
    # kernel = [1, 1, window_width, 1]
    # strides = [1, 1, window_stride, 1]
    # patches = tf.extract_image_patches(image, kernel, strides, [1, 1, 1, 1], 'VALID')
    # patches = tf.transpose(patches, (0, 2, 1, 3))
    # patches = tf.expand_dims(patches, -1)
    patches = images.unfold(3, window_width, window_stride) # (b,c,h,p,window)
    patches = patches.permute((0,1,2,4,3)) # (b,c,h,window,p)

    return patches
