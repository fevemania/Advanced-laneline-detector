import numpy as np
import cv2

import matplotlib.image as mpimg

img_folder = 'assets/'
image = mpimg.imread(img_folder+'bbox-example-image.jpg')

# Define a function that takes an image, a list of bounding boxes, 
# and optional color tuple and line thickness as inputs
# then draws boxes in that color on the output

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    for box in bboxes:
    	cv2.rectangle(draw_img, box[0], box[1], color, thick)
    # return the image copy with boxes drawn
    return draw_img # Change this line to return image copy with boxes
# Add bounding boxes in this format, these are just example coordinates.
bboxes = [((273, 492), (385, 572)),
((475, 505), (550, 560)),
((540, 509), (578, 540)),
((583, 506), (640, 552)),
((636, 501), (679, 542)),
((826, 504), (1140, 684))]

result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()
