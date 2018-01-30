import cv2
import numpy as np

pic = cv2.imread('pic2.png', 0)
lap = cv2.Laplacian(pic, cv2.CV_64F)
lap = cv2.convertScaleAbs(lap, lap)
cv2.imshow('lap', lap)
cv2.waitKey(0)
cv2.destroyAllWindows()

