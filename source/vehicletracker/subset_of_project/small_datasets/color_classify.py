import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import glob
import time

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
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

        # Apply bin_spatial() to get spatial color features
        bin_spatial_features = bin_spatial(image, size=spatial_size)
        # Apply color_hist() to get color histogram features
        color_hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((bin_spatial_features, color_hist_features)))
        
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

spatial = 32
histbin = 32
        
car_features = extract_features(cars, cspace='RGB', spatial_size=(spatial, spatial),
                        hist_bins=histbin, hist_range=(0, 256))
notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(spatial, spatial),
                        hist_bins=histbin, hist_range=(0, 256))

if len(car_features) > 0:

    # Create an array stack of feature vectors
    # X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X = np.concatenate((car_features, notcar_features)).astype(np.float64)

    scaled_X = normalize(X)

    # Create a label array
    y = np.concatenate((np.ones(len(car_features)), 
                   np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    random_state = np.random.randint(0, 100)
    
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=random_state)

    print('Using spatial binning of:', spatial, 'and', histbin, 'histogram bins')
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