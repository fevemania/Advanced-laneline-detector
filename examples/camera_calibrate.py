import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class CameraCalibration():
    def __init__(self, calibration_images, pattern_size=(9,6), retain_calibration_images=False):
        self.camera_matrix = None
        self.distortion_coef = None
        self.calibration_images_success = []
        self.calibration_images_error = []
        self.cal_dist_and_mtx(calibration_images, pattern_size, False)
    
    def __call__(self, img):
        if self.camera_matrix is not None and self.distortion_coef is not None:
            return cv2.undistort(
                    img, self.camera_matrix, self.distortion_coef, None, self.camera_matrix)
        else:
            print("You should calculate Camera Matrix and Distortion coefficient first!")
            return img

    def cal_dist_and_mtx(self, sample_images, pattern_size, retain_calibration_images=False):
        # (x, y) should pass only points where two black and two white squares intersects.
        """
          --------- x-axis ---------->
        : (0,0,0) ...          (8,0,0)
        :
      y-axis
        :
        v (5,0,0) ...          (8,5,0)

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
        for path in sample_images:
            img = mpimg.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Find the chessboard's inner corners
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # If found, add object points, image points
            if found:
                objpoints.append(objp)
                imgpoints.append(corners)
                
                # Draw and display the corners to see what was detected.
                if retain_calibration_images:
                    cv2.drawChessboardCorners(img, pattern_size, corners, found)
                    self.calibration_images_success.append(img)
#            else:
#                if retain_calibration_images:
#                    self.calibration_images_error.append(img)

        img_size = (img.shape[1], img.shape[0])

        if objpoints and imgpoints:
             _, self.camera_matrix, self.distortion_coef, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints, img_size, None, None) 
