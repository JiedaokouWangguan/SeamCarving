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
    num_of_seams_left = num_of_seams
    a = time.time()
    pic = cv2.imread('people.jpg', 1)
    pic1 = cv2.imread('people.jpg', 1)
    while num_of_seams_left > 0:
        rows, cols = pic.shape[:2]

        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        grad = cv2.Laplacian(blur, cv2.CV_64F)
        grad = cv2.convertScaleAbs(grad, grad)
        dp = np.zeros((rows, cols + 2), dtype='int32')
        dp[0:rows, 1:cols + 1] = grad[0:rows, 0:cols]
        dp_rev = np.zeros((rows, cols + 2), dtype='int32')
        dp_rev[0:rows, 1:cols + 1] = grad[0:rows, 0:cols]
        for i in range(0, rows):
            dp[i][0] = 256000
            dp[i][cols + 1] = 256000
            dp_rev[i][0] = 256000
            dp_rev[i][cols + 1] = 256000

        for row in range(1, rows):
            for col in range(1, cols + 1):
                min_value = min(dp[row - 1][col - 1], dp[row - 1][col], dp[row - 1][col + 1])
                r, c = get_result(row, col, min_value, dp)
                dp[row][col] = dp[r][c] + dp[row][col]
        last_row = dp[rows - 1][:]
        index = list(range(0, cols))
        com = list(zip(index, last_row))
        com = sorted(com, key=lambda x: x[1])


        size_to_delete = int(num_of_seams_left * 1)

        delete = np.zeros((rows, size_to_delete))

        delete[:, :] = -1

        delete_index = 0
        com_index = 0
        while delete_index < size_to_delete and com_index < cols:
            last_col = com[com_index][0]

            dp_rev[rows-1][last_col] = 256000
            delete[rows - 1][delete_index] = last_col - 1
            for row in range(rows - 1, 0, -1):
                min_value = min(dp_rev[row - 1][last_col - 1], dp_rev[row - 1][last_col], dp_rev[row - 1][last_col + 1])
                if min_value == 256000:
                    delete[:, delete_index] = -1
                    break

                if dp_rev[row - 1][last_col - 1] == min_value:
                    dp_rev[row - 1][last_col - 1] = 256000
                    last_col = last_col - 1
                elif dp_rev[row - 1][last_col] == min_value:
                    dp_rev[row - 1][last_col] = 256000
                    last_col = last_col
                elif dp_rev[row - 1][last_col + 1] == min_value:
                    dp_rev[row - 1][last_col + 1] = 256000
                    last_col = last_col + 1

                delete[row - 1][delete_index] = last_col - 1

            if delete[rows-1][delete_index] != -1:
                delete_index += 1
                print("index:{}".format(com[com_index][0]))
            com_index += 1

        new_pic = np.zeros((rows, cols + delete_index, 3), dtype='uint8')
        count = 0

        for row in range(0, rows):
            count = 0
            for col in range(0, cols):
                new_pic[row][col + count] = pic[row][col]

                if col in delete[row]:
                    count += 1
                    new_pic[row][col + count] = pic[row][col]

        num_of_seams_left -= delete_index
        pic = new_pic

    cv2.imshow('newPic', pic)
    cv2.imshow('oldPic', pic1)

    b = time.time()
    print(b - a)
    filename = "newPic_{}_{}.png".format(num_of_seams, b-a)
    cv2.imwrite(filename, pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


get_seams(150)

