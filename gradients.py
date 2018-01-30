import cv2
import numpy as np

img = cv2.imread('pic.jpg', 1)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = img.shape[:2]
grad = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# x:
for i in range(0, rows-1):
    for j in range(0, cols-1):
        grad[i][j] = (0.5 *(abs(img_grey[i][j] - img_grey[i+1][j]) + abs(img_grey[i][j] - img_grey[i][j+1])))

for i in range(0, rows):
    grad[i][cols-1] = grad[i][cols-2]
for i in range(0, cols):
    grad[rows-1][i] = grad[rows-2][i]

cv2.imshow('grad', grad)
cv2.waitKey(0)

# edge = cv2.Canny(img_grey, 125, 150)
# cv2.imshow('canny', edge)
# cv2.waitKey(0)
