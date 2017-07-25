import numpy as np
import cv2
from numpy.linalg import inv
import matplotlib.pyplot as plt

# Line Finding Method: Peaks in a Histogram: use peaks to decide explicitly which pixels are part of the lines
def first_frame_lane_finder(bird_eye_binary, M, img, nwindows = 9, debug=False):
    """
    Detect lane pixels and fit to find the lane boundary.
    implement sliding windows and fit a polynomial (only use on the first frame)

    Parameters
    ----------
    bird_eye_binary : Binary image with bird-eye view toward lane
    M               : Perspective transform matrix calculated from cv2.getPerspectiveTransform
    img             : Original frame from camera to be drawed
    nwindows        : The number of sliding windows

    Returns
    -------
    original image with left lane pixels (red), right lane pixels (blue) 
    and whole range drawn in green between left lane and right lane 
    in flat forward-camera view.
    (With small image about its grayscale and color bird-eye-view, also with the measurement values)
    """

    # Take a histogram of the bottom half of the image
    histogram = np.sum(bird_eye_binary[bird_eye_binary.shape[0]//2:, :], axis=0)
    #     plt.plot(histogram)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((bird_eye_binary, bird_eye_binary, bird_eye_binary))*255

    # Find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(bird_eye_binary.shape[0]/nwindows)

    # Identify the x and y positions of all nonzeros pixels in the image
    nonzero = bird_eye_binary.nonzero()
    nonzeroy = nonzero[0]
    nonzerox = nonzero[1]
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step throught the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = bird_eye_binary.shape[0] - (window+1)*window_height
        win_y_high = bird_eye_binary.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 5)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 5)

        # identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If we found > minpix pixls, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Visualize the result
    # Generate x and y values for plotting
    ploty = np.linspace(0, bird_eye_binary.shape[0]-1, bird_eye_binary.shape[0])
    left_fitx = left_fit[0]*(ploty**2) + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*(ploty**2) + right_fit[1]*ploty + right_fit[2]

    car_radius_curve = measure_curvature(left_fitx, right_fitx, ploty)
    car_offset = vehicle_offset(bird_eye_binary.shape[1], left_fitx, right_fitx, ploty)

    warp_zero = np.zeros_like(bird_eye_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    cv2.polylines(out_img, np.int_([pts_left]), False, (255,255,0), thickness=5)
    cv2.polylines(out_img, np.int_([pts_right]), False, (255,255,0), thickness=5)
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    color_warp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    color_warp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    Minv = inv(np.matrix(M))
    
    # Warp the lane onto the warped blank image
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    newwarp[:250, :1280] = (56, 58, 73)

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.4, 0)

    bird_eye_view_drawing = cv2.warpPerspective(result, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    out_img = cv2.resize(out_img, (400, 200))
    bird_eye_view_drawing = cv2.resize(bird_eye_view_drawing, (400, 200))
    rows,cols,channels = out_img.shape
    
    result[25:rows+25, 20:cols+20] = out_img
    result[25:rows+25, cols+40:2*cols+40] = bird_eye_view_drawing

    font = cv2.FONT_HERSHEY_SIMPLEX
    curve_string = "Radius of Curvature = " + str(car_radius_curve) + "(m)"
    cv2.putText(result, curve_string ,(2*cols+60, 50), font, 0.8, (255,255,255), 2, cv2.LINE_AA)

    if car_offset < 0:
        car_offset = np.absolute(car_offset)
        car_offset_string = "Vehicle is " + str(car_offset) + "m left of center."
    else:
        car_offset_string = "Vehicle is " + str(car_offset) + "m right of center."

    cv2.putText(result, car_offset_string ,(2*cols+60, 100), font, 0.8, (255,255,255), 2, cv2.LINE_AA)

    return result, left_fit, right_fit

# It's now much easier to find line pixels once we have used the sliding windows method on the first frame of video!
def frame_lane_finder(bird_eye_binary, prev_left_fit, prev_right_fit, M, img, debug=False):
    """
    Detect lane pixels and fit to find the lane boundary.
    evaluate the (x,y) positions along the left and right lane and fit a polynomial 
    (use on the successor frames)

    Parameters
    ----------
    bird_eye_binary : Binary image with bird-eye view toward lane
    prev_left_fit   : Previous leftfit (the coefficient of quadratic function to evaluate left lane pixel)
    prev_right_fit  : Previous rightfit (the coefficient of quadratic function to evaluate right lane pixel)
    M               : Perspective transform matrix calculated from cv2.getPerspectiveTransform
    img             : Original frame from camera to be drawed

    Returns
    -------
    original image with left lane pixels (red), right lane pixels (blue) 
    and whole range drawn in green between left lane and right lane 
    in flat forward-camera view.
    (With small image about its grayscale and color bird-eye-view, also with the measurement values)
    """

    
    nonzero = bird_eye_binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    
    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] - margin)) & 
                      (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] - margin)) & 
                       (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, bird_eye_binary.shape[0]-1, bird_eye_binary.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    car_radius_curve = measure_curvature(left_fitx, right_fitx, ploty)
    car_offset = vehicle_offset(bird_eye_binary.shape[1], left_fitx, right_fitx, ploty)

    warp_zero = np.zeros_like(bird_eye_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((bird_eye_binary, bird_eye_binary, bird_eye_binary))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    cv2.polylines(out_img, np.int_([pts_left]), False, (255,255,0), thickness=5)
    cv2.polylines(out_img, np.int_([pts_right]), False, (255,255,0), thickness=5)

    out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    color_warp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    color_warp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    Minv = inv(np.matrix(M))
    
    # Warp the lane onto the warped blank image
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    newwarp[:250, :1280] = (56, 58, 73)

    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.4, 0)

    bird_eye_view_drawing = cv2.warpPerspective(result, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    out_img = cv2.resize(out_img, (400, 200))
    bird_eye_view_drawing = cv2.resize(bird_eye_view_drawing, (400, 200))
    rows,cols,channels = out_img.shape
    
    result[25:rows+25, 20:cols+20] = out_img
    result[25:rows+25, cols+40:2*cols+40] = bird_eye_view_drawing

    font = cv2.FONT_HERSHEY_SIMPLEX
    curve_string = "Radius of Curvature = " + str(car_radius_curve) + "(m)"
    cv2.putText(result, curve_string ,(2*cols+60, 50), font, 0.8, (255,255,255), 2, cv2.LINE_AA)

    if car_offset < 0:
        car_offset = np.absolute(car_offset)
        car_offset_string = "Vehicle is " + str(car_offset) + "m left of center."
    else:
        car_offset_string = "Vehicle is " + str(car_offset) + "m right of center."

    cv2.putText(result, car_offset_string ,(2*cols+60, 100), font, 0.8, (255,255,255), 2, cv2.LINE_AA)

    return result, left_fit, right_fit

# Determine the curvature of the lane and vehicle position with respect to center.
def measure_curvature(leftx, rightx, ploty):
    """
    Calculates the deviation of the midpoint of the lane (x_eval)
    from the center of the image (car position)

    Calculates distance to camera in real world coordinate system (e.g. meters), 
    assuming there are 3.7 meters for 700 pixels for x axis.

    Parameters
    ----------
    w      : the width of the image; w/2 is car position
    leftx  : the collection of x values of pixels belongs to left lane boundary
    rightx : the collection of x values of pixels belongs to right lane boundary
    ploty  : the y range from 0...719

    Returns
    -------
    Estimated distance from car to camera in meters.
    if distance > 0, then it means car is at the right of center.
    if distance < 0, then it means car is at the left of center
    """

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720   # meters per pixel in y dimension
    xm_per_pix = 3.7/700  # meters per pixel in x dimension

    # Fit new polynomials to x, y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
                    / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
                     / np.absolute(2*left_fit_cr[0])

    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')
    return int(np.average([left_curverad, right_curverad]))

def vehicle_offset(w, leftx, rightx, ploty):

    """
    Calculates the deviation of the midpoint of the lane (x_eval)
    from the center of the image (car position)

    Calculates distance to camera in real world coordinate system (e.g. meters), 
    assuming there are 3.7 meters for 700 pixels for x axis.

    Parameters
    ----------
    w      : the width of the image; w/2 is car position
    leftx  : the collection of x values of pixels belongs to left lane boundary
    rightx : the collection of x values of pixels belongs to right lane boundary
    ploty  : the y range from 0...719

    Returns
    -------
    Estimated distance from car to camera in meters.
    if distance > 0, then it means car is at the right of center.
    if distance < 0, then it means car is at the left of center
    """
    y_eval = int(np.max(ploty))
    x_eval = int(np.average([leftx[y_eval], rightx[y_eval]])) # the midpoint of the lane
    

    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    return round((w // 2 - x_eval) * xm_per_pix, 3)
