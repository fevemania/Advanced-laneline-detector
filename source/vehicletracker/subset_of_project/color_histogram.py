import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# Read in the image
image = mpimg.imread('same_car_view_2.png')
image = (image*255).astype('uint8')
# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the RGB channels separately
    # 
    # np.histogram() returns a tuple of two arrays, first contains the counts in each of bins
    # 						 second contains the bin edges (so it is one element longer than first one)
    rhist = np.histogram(image[:,:,0], bins=32, range=(0, 256))
    ghist = np.histogram(image[:,:,1], bins=32, range=(0, 256))
    bhist = np.histogram(image[:,:,2], bins=32, range=(0, 256))
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0: len(bin_edges)-1])/2

    # These, collectively, are now our feature vector for this particular cutout image.
    # We can concatenate them in the following way:
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0])) # Concatenate the histograms into a single feature vector
    # Return the individual histograms, bin_centers and feature vector
    return rhist, ghist, bhist, bin_centers, hist_features
    
rh, gh, bh, bincen, feature_vec = color_hist(image, nbins=32, bins_range=(0, 256))

# Plot a figure with all three bar charts
if rh is not None:
    fig = plt.figure(figsize=(12,4))
    plt.subplot(141)
    plt.imshow(image)
    plt.subplot(142)
    plt.bar(bincen, rh[0])
    plt.xlim(0, 256)
    plt.title('R Histogram')
    plt.subplot(143)
    plt.bar(bincen, gh[0])
    plt.xlim(0, 256)
    plt.title('G Histogram')
    plt.subplot(144)
    plt.bar(bincen, bh[0])
    plt.xlim(0, 256)
    plt.title('B Histogram')
    fig.tight_layout()
    plt.show()
else:
    print('Your function is returning None for at least one variable...')
