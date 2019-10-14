import torch

def slide_window(images, window_width, window_stride):
    """
    Input: (batch, c, h, w)
    Output (batch, patches, c, h, w)
    'valid' drops the right-most cols, no padding
    
    Takes (image_height, image_width, 1) input,
    Returns (num_windows, image_height, window_width, 1) output,
    where num_windows is floor((image_width - window_width) / window_stride) + 1
    """

    patches = images.unfold(3, window_width, window_stride) # (b,c,h,p,window)
    patches = patches.permute((0,1,2,4,3)) # (b,c,h,window,p)

    return patches
