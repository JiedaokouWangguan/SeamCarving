import cv2
import numpy as np

img = cv2.imread('pic.jpg', 0)
img1 = cv2.imread('pic.jpg', 1)

# cv2.namedWindow('img', cv2.WINDOW_NORMAL)
# cv2.namedWindow('img1', cv2.WINDOW_NORMAL)

img1grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

cv2.imshow('img1', img1)
cv2.imshow('img1grey', img1grey)

cv2.waitKey(0)
cv2.destroyAllWindows()

