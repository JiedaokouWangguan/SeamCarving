import cv2
import numpy as np
import time


def get_result(row, col, min_value, dp):
    if min_value == dp[row - 1][col - 1]:
        return row -1,col-1
    elif min_value == dp[row - 1][col]:
        return row-1, col
    else:
        return row-1, col + 1


def get_seams(num_of_seams):
    a = time.time()
    pic = cv2.imread('pic2.png', 1)
    pic1 = cv2.imread('pic2.png', 1)
    for count in range(0, num_of_seams):
        rows, cols = pic.shape[:2]

        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        grad = cv2.Laplacian(blur, cv2.CV_64F)
        grad = cv2.convertScaleAbs(grad, grad)
        dp = np.zeros((rows, cols + 2), dtype='int32')
        dp[0:rows, 1:cols + 1] = grad[0:rows, 0:cols]

        dp_rev = np.zeros((rows, cols + 2), dtype='int32')
        dp_rev[0:rows, 1:cols + 1] = grad[0:rows, 0:cols]
        for i in range(0, rows):
            dp[i][0] = 256000
            dp[i][cols + 1] = 256000

        for row in range(1, rows):
            for col in range(1, cols + 1):
                min_value = min(dp[row - 1][col - 1], dp[row - 1][col], dp[row - 1][col + 1])
                r, c = get_result(row, col, min_value, dp)
                dp[row][col] = dp[r][c] + dp[row][col]

        last_row = dp[rows - 1][:]
        index = list(range(0, cols))
        com = list(zip(index, last_row))
        com = sorted(com, key=lambda x: x[1])
        index = com[0][0]

        delete = np.zeros((1, rows))
        delete = delete[0]
        last_col = index
        delete[rows - 1] = last_col-1
        for row in range(rows - 1, 0, -1):
            min_value = min(dp[row - 1][last_col - 1], dp[row - 1][last_col], dp[row - 1][last_col + 1])
            if dp[row - 1][last_col - 1] == min_value:
                last_col = last_col - 1
            elif dp[row - 1][last_col + 1] == min_value:
                last_col = last_col + 1
            delete[row - 1] = last_col-1
        new_pic = np.zeros((rows, cols - 1, 3), dtype='uint8')
        for row in range(0, rows):
            not_visited = True
            for col in range(0, cols):
                if delete[row] == col:
                    not_visited = False
                    continue
                if not_visited:
                    new_pic[row][col] = pic[row][col]
                else:
                    new_pic[row][col-1] = pic[row][col]
        pic = new_pic
    # cv2.imwrite('newPic.png', pic)
    cv2.imshow('newPic', pic)
    cv2.imshow('oldPic', pic1)
    b = time.time()
    print(b - a)
    filename = "oldPic_{}_{}s.png".format(num_of_seams, b-a)
    cv2.imwrite(filename, pic)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


get_seams(200)

