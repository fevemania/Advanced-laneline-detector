import numpy as np
import cv2
import matplotlib.pyplot as plt

def flatten_perspective(img, camera_offset=0, top_offset=70, debug=False):
    """
    Mapping the image from the vehicle front-facing camera to a bird view.
    
    Parameters
    ----------
    img                : Image from the vehicle front-facing camera.
    camera_offset      : The midden x position of the car view - The midden x position of the image
    top_offset         : the offset to control the x value of src vertices with y value equal to roi_y_top.

    Returns
    -------
    Warped image.

    """

    # Get image dimensions
    (h, w) = (img.shape[0], img.shape[1])
    
    roi_y_top = h*0.625
    mid_x = w//2         # midden position of x coordinate in the given image

#     dst_w_offset = 300
    
    # The reason why I use these src and dst vertices is to handle each image with different camera position.
    src_vertices = np.array([[mid_x + top_offset + camera_offset, roi_y_top], [w + 100 + camera_offset, h], [-100 + camera_offset, h], [mid_x - top_offset + camera_offset, roi_y_top]], dtype=np.int32)
    # Define corresponding destination points
    dst_vertices = np.array([[w - 100 + camera_offset, 0], [w - 100 + camera_offset, h], [100 + camera_offset, h], [100 + camera_offset, 0]], dtype=np.int32)
    
#     src_vertices = np.array([[mid_x + offset, roi_y_top], [w - 160, h], 
#                              [200, h], [mid_x - offset, roi_y_top]], dtype=np.int32)
#     dst_vertices = np.array([[w - dst_w_offset, 0], [w - dst_w_offset, h], 
#                              [dst_w_offset, h]    , [dst_w_offset, 0]], dtype=np.int32)
        
    src = np.float32(src_vertices)
    dst = np.float32(dst_vertices)
    
    # perspective transform is a matrix that's returned by the function
    # getPerspectiveTranform
    M = cv2.getPerspectiveTransform(src, dst)
    
    img_size = (img.shape[1], img.shape[0])
    
    # Apply the transform M to the original image to get the wraped image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    if debug == True:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

        # original img with line
        if img.ndim == 2:
            channel = np.uint8(255*img)
            img_with_lines = np.dstack((channel, channel, channel))
        else:
            img_with_lines = img.copy()
            
        pts = src_vertices.reshape((-1,1,2))
        lines = cv2.polylines(img_with_lines, [pts], True, (255,0,0), thickness=3)
        ax1.set_title('img_with_lines')
        ax1.set_axis_off()
        ax1.imshow(img_with_lines)
        
        # flat bird-eye img with line
        if img.ndim == 2:
            channel = np.uint8(255*warped)
            flat_bird_eye_img_with_lines = np.dstack((channel, channel, channel))
        else:
            flat_bird_eye_img_with_lines = warped.copy()
        pts = dst_vertices.reshape((-1,1,2))
        lines = cv2.polylines(flat_bird_eye_img_with_lines, [pts], True, (255,0,0), thickness=3)
        ax2.set_title('flat_bird_eye_img_with_lines')
        ax2.set_axis_off()
        ax2.imshow(flat_bird_eye_img_with_lines)
        plt.show()

    return warped, M