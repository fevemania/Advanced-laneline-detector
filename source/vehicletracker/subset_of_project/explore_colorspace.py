# It looks like cars are more saturated in color, while the background is generally pale.

# Why to add star

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

def plot3d(pixels, color_space, colors_rgb, axis_labels=list("RGB"), 
			axis_limits=[(0, 255), (0, 255), (0, 255)]):
	"""Plot pixels in 3D."""
	# Convert subsampled image to desired color space(s)
	if color_space != 'RGB':
		if  color_space == 'HSV':
		    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
		    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
		    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
		    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
		    pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2YCrCb)

	# Create figure and 3D axes
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection='3d')

	# Set axis limits
	ax.set_xlim(*axis_limits[0])
	ax.set_ylim(*axis_limits[1])
	ax.set_zlim(*axis_limits[2])

	# Set axis labels and sizes
	ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
	ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
	ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
	ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

	# # Plot pixel values with colors given in colors_rgb
	ax.scatter(
		pixels[:,:,0].ravel(),
		pixels[:,:,1].ravel(),
		pixels[:,:,2].ravel(),
		c=colors_rgb.reshape((-1,3)), edgecolors='none')

	return ax # return Axes3D object for further manipulation


def main():
	img_folder = 'assets/'
	# Read a color image
	# img_name = '001240.png'
	img_name = '6.png'
	# img_name = 'test_img.jpg'
	# img = mpimg.imread(img_folder+img_name)
	img = cv2.cvtColor(cv2.imread(img_folder+img_name), cv2.COLOR_BGR2RGB)

	# Select a small fraction of pixels to plot by subsampling it
	scale = max(img.shape[0], img.shape[1], 64) / 64 # at most 64 rows and columns
	img_small_RGB = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
	img_small_rgb = img_small_RGB / 255. # scaled to [0, 1], only for plotting

	# Plot and show
	color_space = 'HSV'
	plot3d(img_small_RGB, color_space, img_small_rgb, axis_labels=list(color_space))
	plt.show()

if __name__ == '__main__':
	main()
