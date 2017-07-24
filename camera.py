import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280) 
cap.set(4, 720)

while (True):
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
