import cv2
import matplotlib.pyplot as plt
import numpy as np


def hls_select(img, channel='S', thresh=(0, 255), debug=False):
    """
    Apply threshold on image with hls colorspace.
    
    Parameters
    ----------
    img          : RGB image.
    channel      : Select channel to apply threshold on. 
                   If channel is 'HLS' then don't apply threshold.
    thresh       : Threshold range between two particular pixel values. 
                   The range is only useful when channel is not 'HLS' or 'hls'.
    debug        : Flag indicating if we need to display the output.
                   
    Returns
    ------
    None          : If channel == 'HLS', 'hls'
    binary_output : Binary image if channel == 'H', 'L', 'S', 'h', 'l', 's'
    """

    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    channel = channel.upper()
    
    if channel == 'H':
        channel_img = hls[:, :, 0]
    elif channel == 'L':
        channel_img = hls[:, :, 1]
    elif channel == 'S':
        channel_img = hls[:, :, 2]
    elif channel == 'HLS' and debug == True:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))
        H = hls[:,:,0]
        L = hls[:,:,1]
        S = hls[:,:,2]

        axes[0].set_title('H')
        axes[0].set_axis_off()
        axes[1].set_title('L')
        axes[1].set_axis_off()
        axes[2].set_title('S')
        axes[2].set_axis_off()
        axes[0].imshow(H, cmap='gray')
        axes[1].imshow(L, cmap='gray')
        axes[2].imshow(S, cmap='gray')
        return None
    else:
        return None
    
    binary_output = np.zeros_like(channel_img)
    binary_output[(channel_img > thresh[0]) & (channel_img <= thresh[1])] = 1
    
    if debug == True:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))
        
        axes[0].set_title('original image')
        axes[0].set_axis_off()
        axes[0].imshow(img)        
        
        axes[1].set_title(channel + ' original')
        axes[1].set_axis_off()
        axes[1].imshow(channel_img, cmap='gray')

        axes[2].set_title(channel + ' with_range_select')
        axes[2].set_axis_off()
        axes[2].imshow(binary_output, cmap='gray')
        plt.show()
        
    return binary_output

def rgb_select(img, channel='R', thresh=(0, 255), debug=False):
    """
    Apply threshold on image with RGB colorspace.
    
    Parameters
    ----------
    img          : RGB image.
    channel      : Select channel to apply threshold on. 
                   If channel is 'RGB', then don't apply threshold.
    thresh       : Threshold range between two particular pixel values. 
                   The range is only useful when channel is not 'RGB' or 'RGB'.
    debug        : Flag indicating if we need to display the output.
                   
    Returns
    -------
    None          : if channel == 'RGB', 'rgb'.
    binary_output : Binary image if channel == 'R', 'G', 'B', 'r', 'g', 'b'
    """
    rgb = img.copy()
    channel = channel.upper()
    
    if channel == 'R':
        channel_img = rgb[:, :, 0]
    elif channel == 'G':
        channel_img = rgb[:, :, 1]
    elif channel == 'B':
        channel_img = rgb[:, :, 2]
    elif channel == 'RGB' and debug == True:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))
        R = rgb[:,:,0]
        G = rgb[:,:,1]
        B = rgb[:,:,2]

        axes[0].set_title('R')
        axes[0].set_axis_off()
        axes[1].set_title('B')
        axes[1].set_axis_off()
        axes[2].set_title('G')
        axes[2].set_axis_off()
        axes[0].imshow(R, cmap='gray')
        axes[1].imshow(G, cmap='gray')
        axes[2].imshow(B, cmap='gray')
        plt.show()
        return None
    else:
        return None
    
    binary_output = np.zeros_like(channel_img)
    binary_output[(channel_img > thresh[0]) & (channel_img <= thresh[1])] = 1
    
    if debug == True:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3))
        
        axes[0].set_title('original image')
        axes[0].set_axis_off()
        axes[0].imshow(img)        
        
        axes[1].set_title(channel + ' original')
        axes[1].set_axis_off()
        axes[1].imshow(channel_img, cmap='gray')

        axes[2].set_title(channel + ' with_range_select')
        axes[2].set_axis_off()
        axes[2].imshow(binary_output, cmap='gray')
        plt.show()
        
    return binary_output