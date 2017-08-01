import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
from skimage.feature import hog

# Read in our vehicles and non-vehicles
#images = glob.glob('*.jpeg')
#cars = []
#notcars = []

#for image in images:
#    if 'image' in image or 'extra' in image:
#        notcars.append(image)
#    else:
#        cars.append(image)
        
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        # The visualise=True flag tells the function to output a visualization of the HOG feature computation as well, 
        # which we're calling hog_image in this case.
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=feature_vec, block_norm  = 'L1-sqrt')
        return features, hog_image
    else:
        # Use skimage.hog() to get features only      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec, block_norm  = 'L1-sqrt')
        return features

# Generate a random index to look at a car image
# ind = np.random.randint(0, len(cars))
# Read in the image
#image = mpimg.imread(cars[ind])
image_RGB = mpimg.imread('lego.jpg')

image_YCrCb = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2YCrCb)
orient = 10 
pix_per_cell = 8 
cell_per_block = 2

_, hog_image_R = get_hog_features(image_RGB[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
_, hog_image_G = get_hog_features(image_RGB[:,:,1], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
_, hog_image_B = get_hog_features(image_RGB[:,:,2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

fig = plt.figure(figsize=(8,4))
plt.subplot(241)
plt.imshow(image_RGB)
plt.title('RGB')
plt.subplot(242)
plt.imshow(hog_image_R, cmap='gray')
plt.title('R')
plt.subplot(243)
plt.imshow(hog_image_G, cmap='gray')
plt.title('G')
plt.subplot(244)
plt.imshow(hog_image_B, cmap='gray')
plt.title('B')

_, hog_image_Y = get_hog_features(image_YCrCb[:,:,0], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
_, hog_image_Cr = get_hog_features(image_YCrCb[:,:,1], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)
_, hog_image_Cb = get_hog_features(image_YCrCb[:,:,2], orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)

plt.subplot(245)
plt.imshow(image_YCrCb)
plt.title('YCrCb')
plt.subplot(246)
plt.imshow(hog_image_Y, cmap='gray')
plt.title('Y')
plt.subplot(247)
plt.imshow(hog_image_Cr, cmap='gray')
plt.title('Cr')
plt.subplot(248)
plt.imshow(hog_image_Cb, cmap='gray')
plt.title('Cb')
fig.tight_layout()
plt.show()

# # Define HOG parameters
# # Call our function with vis=True to see an image output
# #features, hog_image = get_hog_features(gray, orient, pix_per_cell, cell_per_block, vis=True, feature_vec=False)


# # Plot the examples

# #plt.subplot(122)
# #plt.imshow(hog_image)
# #plt.title('HOG Visualization')

# #plt.show()
