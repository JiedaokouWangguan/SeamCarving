import cv2
import numpy as np

img = np.zeros((20, 20), dtype=np.int8)
print(img)
print("------------")
con = np.array( [[1, 1], [1, 2], [1, 3], [1, 4], [2, 4], [3, 4], [3, 3], [3, 2], [3, 1], [2, 1]])
# cv2.fillConvexPoly(img, con, 1)

c = 0
rows = 20
cols = 20

for row in range(rows):
    for col in range(cols):
        if [row, col] in con:
            img[row][col] = 1
print(img)

# print(c)



