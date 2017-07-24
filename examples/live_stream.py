from camera_calibrate import CameraCalibration
import glob
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from color_thresh import hls_select
from color_thresh import rgb_select
from gradient_thresh import abs_sobel_thresh
from gradient_thresh import magnitude_thresh
from gradient_thresh import direction_threshold
from lane_finder import first_frame_lane_finder
from lane_finder import frame_lane_finder
from perspective import flatten_perspective

#cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)

cap = cv2.VideoCapture(0)

#def static_var(varname, value):
#    def decorate(func):
#        setattr(func, varname, value)
#        return func
#    return decorate
#
#@static_var("counter", 0)
#@static_var("left_fit", None)
#@static_var("right_fit", None)
#def pipeline(img):
#    pipeline.counter += 1
##     img = cal(img)
#
#    r_binary = rgb_select(img, channel='R', thresh=(215, 255), debug=False)
#    s_binary = hls_select(img, thresh=(70, 235), debug=False)
#    gradx_binary = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(20, 150), debug=False)
#    h_binary = hls_select(img, channel='h', thresh=(19, 70), debug=False)
#    
#    combined_binary = np.zeros_like(gradx_binary)
#    combined_binary[(gradx_binary == 1) | (((h_binary == 1) & (s_binary == 1)) | (r_binary == 1))] = 1
#    out_img = np.dstack((combined_binary, combined_binary, combined_binary)) * 255
#    
#    
#    bird_eye_binary, M = flatten_perspective(combined_binary, camera_offset=-30, half_of_lane_width=125 , debug=False)

    #if pipeline.counter == 1:
    #    out_img, pipeline.left_fit, pipeline.right_fit = first_frame_lane_finder(bird_eye_binary, M, img, debug=True)
    #else:
    #    out_img, pipeline.left_fit, pipeline.right_fit = frame_lane_finder(bird_eye_binary, 
    #                                             pipeline.left_fit, pipeline.right_fit, M, img, debug=True)
    #return out_img.astype(np.uint8)
#    return np.dstack((bird_eye_binary, bird_eye_binary, bird_eye_binary))*255


#while (cap.isOpened()):
#    print('here')
#    _, frame = cap.read()
    #frame = cv2.cvtColor(cv2.COLOR_BGR2RGB)
    #out_img = pipeline(frame)
    #out_img = cv2.cvtColor(cv2.COLOR_RGB2BGR)
    #cv2.imshow('frame', out_img)
#    cv2.imshow('frame', frame)
#    if cv2.waitKey(33) & 0xFF == ord('q'):
#    break
cap.release()
cv2.destroyAllWindows()
