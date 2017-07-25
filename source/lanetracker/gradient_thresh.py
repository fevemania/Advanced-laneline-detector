import cv2
import numpy as np
import matplotlib.pyplot as plt 

def abs_sobel_thresh(single_channel_img, orient='x', sobel_kernel=3, thresh=(0, 255), debug=False):
    """
    Calculate absolute gradient value with respect to x direction or y direction.

    Parameters
    ----------
    single_channel_img          : the single channel image to thresh (usually grayscale or s channel image)
    orient       : Specify the direction to thresh with sobel operator. Either 'x' or 'y'.
    sobel_kernel : Kernel size of sobel kernel
    thresh       : Threshold range between two particular pixel values.
    debug        : Flag indicating if we need to display the output.

    Returns
    -------
    binary output: Image which has value 1 if there is the gredient value between two particular thresh values
                                            othervwise has value 0.
    """
    
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(single_channel_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(single_channel_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    if debug == True:
        plt.imshow(binary_output, cmap='gray')
        plt.show()
    
    return binary_output

def magnitude_thresh(single_channel_img, sobel_kernel=3, thresh=(0, 255), debug=False):
    """
    Calculate gradient magnitude.

    Parameters
    ----------
    single_channel_img          : the single channel image to thresh (usually grayscale or s channel image)
    sobel_kernel : Kernel size of sobel kernel
    thresh       : Threshold range between two particular pixel values.
    debug        : Flag indicating if we need to display the output.

    Returns
    -------
    binary output: Image which has value 1 if there is the gredient value between two particular thresh values
                                            othervwise has value 0.    
    """    
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(single_channel_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(single_channel_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gredient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    gradmag = (255*gradmag/np.max(gradmag)).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    
    if debug == True:
        plt.imshow(binary_output, cmap='gray')
        plt.show()
    
    return binary_output

def direction_threshold(single_channel_img, sobel_kernel=3, thresh=(0, np.pi/2), debug=False):
    """
    Calculate gradient direction.

    Parameters
    ----------
    single_channel_img          : the single channel image to thresh (usually grayscale or s channel image)
    sobel_kernel : Kernel size of sobel kernel
    thresh       : Threshold range between two particular pixel values.
    debug        : Flag indicating if we need to display the output.    

    Returns
    -------
    binary output: Image which has value 1 if there is the gredient value between two particular thresh values
                                            othervwise has value 0.    
    """
    sobelx = cv2.Sobel(single_channel_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(single_channel_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    
    if debug == True:
        plt.imshow(binary_output, cmap='gray')
        plt.show()    
    
    return binary_output
