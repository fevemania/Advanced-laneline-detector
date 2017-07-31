import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import glob
import time

from skimage.feature import hog

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, block_norm, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        # The visualise=True flag tells the function to output a visualization of the HOG feature computation as well, 
        # which we're calling hog_image in this case.
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm=block_norm, transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    else:
        # Use skimage.hog() to get features only      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm=block_norm, transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, block_norm='L1', hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in imgs:
        # Read in each one by one
        image = mpimg.imread(img)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(image.shape[2]):
                hog_features.append(get_hog_features(image[:,:,channel], orient, pix_per_cell, cell_per_block, block_norm,vis=False, feature_vec=True))
                
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, block_norm, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)
        
    # Return list of feature vectors
    return features

def normalize(dataset, vis=False):
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(dataset)
    # Apply the scaler to dataset
    scaled_X = X_scaler.transform(dataset)

    # Plot random car image to show the effect of normalization.
    if vis == True:
        car_ind = np.random.randint(0, len(cars))

        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.show()

    return scaled_X

images = glob.glob('*.jpeg')
cars = []
notcars = []
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

# Reduce the sample size because HOG features are slow to compute

sample_size = 500
cars = cars[:sample_size]
notcars = notcars[:sample_size]

print("Number of cars sample:", len(cars))
print("Number of notcars sample:", len(notcars))

### TODO: Tweak these parameters and see how the results change.
colorspace = 'RGB'
orient = 12
pix_per_cell = 4  # this parameter effect siginifantly on features number, if pix_per_cell half as much as it was , features number times 4
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or 'ALL'
block_norm = 'L1-sqrt'

t = time.time()        
car_features = extract_features(cars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, 
    cell_per_block=cell_per_block, block_norm=block_norm, hog_channel=hog_channel)
notcar_features = extract_features(notcars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, 
    cell_per_block=cell_per_block, block_norm=block_norm, hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features')

if len(car_features) > 0:

    # Create an array stack of feature vectors
    X = np.concatenate((car_features, notcar_features)).astype(np.float64)

    scaled_X = normalize(X)

    # Create a label array
    y = np.concatenate((np.ones(len(car_features)), 
                   np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    random_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=random_state)

    print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    svc = LinearSVC()
    # Check the training time for the LinearSVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train LinearSVC...')

    print('Acc of LinearSVC = ', round(svc.score(X_test, y_test), 4))
    
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My LinearSVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict, 'labels with LinearSVC')
else: 
    print('Your function only returns empty feature vectors...')