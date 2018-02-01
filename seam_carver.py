import cv2
import numpy as np
import time


class SeamCarver(object):
    def __init__(self, src, dst, rate):
        self.src = src
        self.dst = dst
        self.rate = rate

    def set_pic(self, src, dst):
        self.src = src
        self.dst = dst

    def get_rows_cols(self):
        pic = cv2.imread(self.src, 0)
        rows, cols = pic.shape[:2]
        return rows, cols

    @staticmethod
    def get_invalid_value(pic):
        rows, cols = pic.shape[:2]
        return 255 * rows if rows > cols else 255 * cols

    def carve(self, tgt_height, tgt_width):
        """
        :param tgt_height:target height
        :param tgt_width:target width
        :return:
        """
        time_begin = time.time()
        pic = cv2.imread(self.src, 1)
        rows, cols = pic.shape[:2]
        diff_rows = tgt_height - rows
        diff_cols = tgt_width - cols

        invalid_value = self.get_invalid_value(pic)

        while diff_rows != 0 or diff_cols != 0:
            rows, cols = pic.shape[:2]
            gradient = self.laplacian(pic, True, 3)
            dp_hori = np.array([])
            dp_hori_back = np.array([])
            dp_verti = np.array([])
            dp_verti_back = np.array([])
            if diff_rows != 0:
                dp_hori, dp_hori_back = self.dp_horizontal(gradient, invalid_value)

            if diff_cols != 0:
                dp_verti, dp_verti_back = self.dp_vertical(gradient, invalid_value)

            cost_hori = -1
            cost_verti = -1
            rows_modify = []
            cols_modify = []
            index_rows_modify = 0
            index_cols_modify = 0

            if dp_hori.any():
                last_col = dp_hori[:, -1]
                index = np.arange(0, rows)
                tuples = list(zip(index, last_col))
                sorted_tuples = sorted(tuples, key=lambda x: x[1])
                num_rows_delete = abs(int(diff_rows * self.rate)) + 1
                rows_modify = np.zeros((num_rows_delete, cols), dtype='int16')
                rows_modify[:, :] = -1
                index_sorted_tuples = 0

                while index_rows_modify < num_rows_delete and index_sorted_tuples < rows:
                    index_cur_row = int(sorted_tuples[index_sorted_tuples][0])
                    rows_modify[index_rows_modify][cols - 1] = index_cur_row - 1
                    # index_cur_row is 1-based while row_delete is 0-based

                    for col in range(cols - 1, 0, -1):
                        min_value = min(dp_hori_back[index_cur_row - 1][col - 1], dp_hori_back[index_cur_row][col - 1],
                                        dp_hori_back[index_cur_row + 1][col - 1])
                        if min_value == invalid_value:
                            rows_modify[index_rows_modify, :] = -1
                            break

                        if dp_hori_back[index_cur_row - 1][col - 1] == min_value:
                            dp_hori_back[index_cur_row - 1][col - 1] = invalid_value
                            index_cur_row = index_cur_row - 1
                        elif dp_hori_back[index_cur_row][col - 1] == min_value:
                            dp_hori_back[index_cur_row][col - 1] = invalid_value
                            index_cur_row = index_cur_row
                        elif dp_hori_back[index_cur_row + 1][col - 1] == min_value:
                            dp_hori_back[index_cur_row + 1][col - 1] = invalid_value
                            index_cur_row = index_cur_row + 1

                        rows_modify[index_rows_modify][
                            col - 1] = index_cur_row - 1  # index_cur_row is 1-based while row_delete is 0-based

                    if rows_modify[index_rows_modify][cols - 1] != -1:
                        index_rows_modify += 1

                    index_sorted_tuples += 1
                cost_hori = 0
                for i in range(0, index_rows_modify):
                    cost_hori += last_col[rows_modify[i][cols - 1] + 1]

            if dp_verti.any():
                last_row = dp_verti[-1, :]
                index = np.arange(0, cols)
                tuples = list(zip(index, last_row))
                sorted_tuples = sorted(tuples, key=lambda x: x[1])
                num_cols_delete = abs(int(diff_cols * self.rate)) + 1
                cols_modify = np.zeros((rows, num_cols_delete), dtype='int16')
                cols_modify[:, :] = -1
                index_sorted_tuples = 0

                while index_cols_modify < num_cols_delete and index_sorted_tuples < cols:
                    index_cur_col = int(sorted_tuples[index_sorted_tuples][0])
                    cols_modify[rows - 1][index_cols_modify] = index_cur_col - 1
                    # index_cur_col is 1-based while col_delete is 0-based

                    for row in range(rows - 1, 0, -1):
                        min_value = min(dp_verti_back[row - 1][index_cur_col - 1],
                                        dp_verti_back[row - 1][index_cur_col],
                                        dp_verti_back[row - 1][index_cur_col + 1])
                        if min_value == invalid_value:
                            cols_modify[:, index_cols_modify] = -1
                            break

                        if dp_verti_back[row - 1][index_cur_col - 1] == min_value:
                            dp_verti_back[row - 1][index_cur_col - 1] = invalid_value
                            index_cur_col = index_cur_col - 1
                        elif dp_verti_back[row - 1][index_cur_col] == min_value:
                            dp_verti_back[row - 1][index_cur_col] = invalid_value
                            index_cur_col = index_cur_col
                        elif dp_verti_back[row - 1][index_cur_col + 1] == min_value:
                            dp_verti_back[row - 1][index_cur_col + 1] = invalid_value
                            index_cur_col = index_cur_col + 1

                        cols_modify[row - 1][
                            index_cols_modify] = index_cur_col - 1  # index_cur_col is 1-based while col_delete is 0-based

                    if cols_modify[rows - 1][index_cols_modify] != -1:
                        index_cols_modify += 1

                    index_sorted_tuples += 1
                cost_verti = 0
                for i in range(0, index_cols_modify):
                    cost_verti += last_row[cols_modify[rows-1][i] + 1]

            if cost_verti != -1 and cost_hori != -1:
                if cost_hori < cost_verti:
                    pic, diff_rows = self.update_seams_hori(pic, rows_modify, index_rows_modify, diff_rows)
                else:
                    pic, diff_cols = self.update_seams_verti(pic, cols_modify, index_cols_modify, diff_cols)

            elif cost_verti != -1:
                pic, diff_cols = self.update_seams_verti(pic, cols_modify, index_cols_modify, diff_cols)
            else:
                # cost_hori != -1
                pic, diff_rows = self.update_seams_hori(pic, rows_modify, index_rows_modify, diff_rows)

        cv2.imwrite(self.dst, pic)
        time_end = time.time()
        time_consumed = time_end - time_begin
        print("new picture:{}, size:{}x{}, time consumed:{}".format(self.dst, tgt_height, tgt_width, time_consumed))
        return pic

    @staticmethod
    def update_seams_hori(pic, rows_modify, index_rows_delete, diff_rows):
        rows, cols = pic.shape[:2]
        if diff_rows < 0:
            # remove seams
            diff_new = diff_rows + index_rows_delete
            new_pic = np.zeros((rows - index_rows_delete, cols, 3), dtype='uint8')
            for col in range(0, cols):
                count = 0
                for row in range(0, rows):
                    if row in rows_modify[:, col]:
                        count += 1
                    else:
                        new_pic[row - count][col] = pic[row][col]
        else:
            # insert seams
            diff_new = diff_rows - index_rows_delete
            new_pic = np.zeros((rows + index_rows_delete, cols, 3), dtype='uint8')
            for col in range(0, cols):
                count = 0
                for row in range(0, rows):
                    new_pic[row + count][col] = pic[row][col]

                    if row in rows_modify[:, col]:
                        count += 1
                        new_pic[row + count][col] = pic[row][col]
        return new_pic, diff_new

    @staticmethod
    def update_seams_verti(pic, col_delete, index_cols_delete, diff_cols):
        rows, cols = pic.shape[:2]
        if diff_cols < 0:
            # remove seams
            diff_new = diff_cols + index_cols_delete
            new_pic = np.zeros((rows, cols - index_cols_delete, 3), dtype='uint8')
            for row in range(0, rows):
                count = 0
                for col in range(0, cols):
                    if col in col_delete[row]:
                        count += 1
                    else:
                        new_pic[row][col - count] = pic[row][col]
        else:
            # insert seams
            diff_new = diff_cols - index_cols_delete
            new_pic = np.zeros((rows, cols + index_cols_delete, 3), dtype='uint8')
            for row in range(0, rows):
                count = 0
                for col in range(0, cols):
                    if col + count == 588:
                        sdf = 1
                    new_pic[row][col + count] = pic[row][col]

                    if col in col_delete[row]:
                        count += 1
                        new_pic[row][col + count] = pic[row][col]
        return new_pic, diff_new

    @staticmethod
    def dp_horizontal(pic, invalid_value):
        # use dp to calculate the energy for each row
        rows, cols = pic.shape[:2]
        dp = np.zeros((rows + 2, cols), dtype='int32')
        dp[1:rows + 1, 0:cols] = pic[0:rows, 0:cols]
        dp_back = np.zeros((rows + 2, cols), dtype='int32')
        dp_back[1:rows + 1, 0:cols] = pic[0:rows, 0:cols]
        for i in range(0, cols):
            dp[0][i] = invalid_value
            dp[rows + 1][i] = invalid_value
            dp_back[0][i] = invalid_value
            dp_back[rows + 1][i] = invalid_value
        for col in range(1, cols):
            for row in range(1, rows + 1):
                min_value = min(dp[row - 1][col - 1], dp[row][col - 1], dp[row + 1][col - 1])
                if min_value == dp[row - 1][col - 1]:
                    dp[row][col] = dp[row - 1][col - 1] + dp[row][col]
                elif min_value == dp[row][col - 1]:
                    dp[row][col] = dp[row][col - 1] + dp[row][col]
                else:
                    dp[row][col] = dp[row + 1][col - 1] + dp[row][col]
        return dp, dp_back

    @staticmethod
    def dp_vertical(pic, invalid_value):
        # use dp to calculate the energy for each column
        rows, cols = pic.shape[:2]
        dp = np.zeros((rows, cols + 2), dtype='int32')
        dp[0:rows, 1:cols + 1] = pic[0:rows, 0:cols]
        dp_back = np.zeros((rows, cols + 2), dtype='int32')
        dp_back[0:rows, 1:cols + 1] = pic[0:rows, 0:cols]
        for i in range(0, rows):
            dp[i][0] = invalid_value
            dp[i][cols + 1] = invalid_value
            dp_back[i][0] = invalid_value
            dp_back[i][cols + 1] = invalid_value
        for row in range(1, rows):
            for col in range(1, cols + 1):
                min_value = min(dp[row - 1][col - 1], dp[row - 1][col], dp[row - 1][col + 1])
                if min_value == dp[row - 1][col - 1]:
                    dp[row][col] = dp[row - 1][col - 1] + dp[row][col]
                elif min_value == dp[row - 1][col]:
                    dp[row][col] = dp[row - 1][col] + dp[row][col]
                else:
                    dp[row][col] = dp[row - 1][col + 1] + dp[row][col]
        return dp, dp_back

    @staticmethod
    def laplacian(pic, blur, gaussian_ksize):
        # use Laplacian operator
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        if blur:
            gray = cv2.GaussianBlur(gray, (gaussian_ksize, gaussian_ksize), 0)
        grad = cv2.Laplacian(gray, cv2.CV_64F)
        grad = cv2.convertScaleAbs(grad, grad)
        return grad

    @staticmethod
    def sobel(pic, ksize):
        # use Sobel operator
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=ksize)
        grad = cv2.convertScaleAbs(grad, grad)
        return grad
