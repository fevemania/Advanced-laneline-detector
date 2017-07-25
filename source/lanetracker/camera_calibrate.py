# Note: This module code is based on the repo "detecting-road-features" written by navoshta.
# https://github.com/navoshta/detecting-road-features/blob/master/source/lanetracker/camera.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CameraCalibration(object):
    """
    Implement camera calibration based on the set of calibration images 
    which may be achieved by using an object with a known geometry and easily detectable feature points. 
    Such an object is called a calibration rig or calibration pattern, 
    and OpenCV has built-in support for a chessboard as a calibration rig

    """

    def __init__(self, calibration_patterns, pattern_size=(9,6), retain_calibration_patterns=False):
        """
        Initializes parameters based on the set of calibration patterns.

        Parameters
        ----------
        calibration_patterns         : A set of images used to calculate camera_matrix and distortion coefficient.
                                       (usually used chessboard image)
        pattern_size                 : Shape of the calibration patterns, format=(col, row)
        retain_calibration_patterns  : Flag indicating if we need to preserve calibration patterns.
        """
        self.camera_matrix = None
        self.distortion_coef = None
        self.calibration_patterns_success = []
        self.calibration_patterns_error = []
        self.cal_dist_and_mtx(calibration_patterns, pattern_size, retain_calibration_patterns)
    
    def __call__(self, img):
        """
        Calibrates an image based on saved settings.
    
        Parameters
        ----------
        img:        Image to calibrate.

        Returns
        -------
        Calibrated image.
        """
        if self.camera_matrix is not None and self.distortion_coef is not None:
            return cv2.undistort(
                    img, self.camera_matrix, self.distortion_coef, None, self.camera_matrix)
        else:
            print("You should calculate Camera Matrix and Distortion coefficient first!")
            return img

    def cal_dist_and_mtx(self, calibration_patterns, pattern_size, retain_calibration_patterns):
        """
        calculate camera matrix and distortion coefficient based on a set of calibration patterns.

        Parameters
        ----------
        the same as __init__ function.
        

        coordinate system of pattern image: 
          --------- x-axis ---------->
        : (0,0,0) ...          (8,0,0)
        :
      y-axis
        :
        v (5,0,0) ...          (8,5,0)
        
        (x, y) should pass only points where two black and two white squares intersects.

        """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0), ....,(8,5,0)
        # use numpy mgrid function to generate the coordinates values for a given grid size.
        grid_x, grid_y = pattern_size[0], pattern_size[1]
        objp = np.zeros((grid_y * grid_x, 3), np.float32)
        objp[:,:2] = np.mgrid[:grid_x, :grid_y].T.reshape(-1,2)

        # Arrays to store object points and image points from all the sample images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.
        
        # Step through the list and search for chessboard corners in distorted calibration images.
        for path in calibration_patterns:
            img = mpimg.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard's inner corners
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # If found, add object points, image points
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
                
                # Draw and display the corners to see what was detected.
                if retain_calibration_patterns:
                    cv2.drawChessboardCorners(img, pattern_size, corners, found)
                    self.calibration_patterns_success.append(img)
            else:
                if retain_calibration_patterns:
                    self.calibration_patterns_error.append(img)

            img_size = (img.shape[1], img.shape[0])

        if objpoints and imgpoints:
             _, self.camera_matrix, self.distortion_coef, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None)