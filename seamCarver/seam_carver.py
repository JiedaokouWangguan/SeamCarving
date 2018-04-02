import cv2
import numpy as np
import time


class SeamCarver(object):
    def __init__(self, src, dst, rate_insert, rate_remove):
        self.src = src
        self.dst = dst
        if rate_remove <= 0.0 or rate_remove >= 1.0 or rate_insert <= 0.0 or rate_insert >= 1.0:
            print("The value of rate is illegal. Legal value:(0, 1)")
            return
        self.rate_insert = rate_insert
        self.rate_remove = rate_remove
        self.points = []
        self.drawing = False
        self.tmp_pic = []

    def set_pic(self, src, dst):
        self.src = src
        self.dst = dst

    def set_rate(self, rate_insert, rate_remove):
        self.rate_insert = rate_insert
        self.rate_remove = rate_remove

    def get_rows_cols(self):
        pic = cv2.imread(self.src, 0)
        rows, cols = pic.shape[:2]
        return rows, cols

    @staticmethod
    def get_invalid_value(pic):
        rows, cols = pic.shape[:2]
        return 255 * rows if rows > cols else 255 * cols

# object removal
    def remove_obj(self, rm_row, rm_col):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_circle_remove_obj)
        self.tmp_pic = cv2.imread(self.src, 1)
        while 1:
            cv2.imshow('image', self.tmp_pic)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('a'):
                break

        self.__remove_obj(rm_row, rm_col)
        pass

    def draw_circle_remove_obj(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            for i in range(-5, 5):
                for j in range(-5, 5):
                    self.points.append((x+i, y+j))
                    self.tmp_pic[y+j][x+i][0] = 0
                    self.tmp_pic[y+j][x+i][1] = 0
                    self.tmp_pic[y+j][x+i][2] = 0

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                for i in range(-5, 5):
                    for j in range(-5, 5):
                        self.points.append((x+i, y+j))
                        self.tmp_pic[y+j][x+i][0] = 0
                        self.tmp_pic[y+j][x+i][1] = 0
                        self.tmp_pic[y+j][x+i][2] = 0

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    @staticmethod
    def __num_points(mask):
        num_p = 0
        for row in mask:
            for element in row:
                if element == 1:
                    num_p += 1
        print(num_p)
        return num_p

    def __remove_obj(self, rm_row, rm_col):
        points = self.points
        time_begin = time.time()
        pic = cv2.imread(self.src, 1)
        if pic is None:
            print("can't open file:{}.".format(self.src))
            return
        rows, cols = pic.shape[:2]

        invalid_value = self.get_invalid_value(pic)
        mask = np.zeros((rows, cols), dtype='int8')
        for point in points:
            x = point[0]
            y = point[1]
            mask[y][x] = 1

        while self.__num_points(mask) != 0:
            rows, cols = pic.shape[:2]
            dp_hori = np.array([])
            dp_hori_back = np.array([])
            dp_verti = np.array([])
            dp_verti_back = np.array([])
            if rm_row:
                gradient_hori = self.sobel(pic, 3)
                # mask
                dp_hori, dp_hori_back = self.dp_horizontal_rm_obj(gradient_hori, invalid_value, mask)

            if rm_col:
                gradient_verti = self.sobel(pic, 3)
                # mask
                dp_verti, dp_verti_back = self.dp_vertical_rm_obj(gradient_verti, invalid_value, mask)

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
                num_rows_delete = 1
                rows_modify = np.zeros((num_rows_delete, cols), dtype='int32')
                rows_modify[:, :] = -1
                index_sorted_tuples = 0

                while index_rows_modify < num_rows_delete and index_sorted_tuples < rows:
                    index_cur_row = int(sorted_tuples[index_sorted_tuples][0])
                    rows_modify[index_rows_modify][cols - 1] = index_cur_row - 1
                    # index_cur_row is 1-based while row_delete is 0-based

                    for col in range(cols - 1, 0, -1):
                        min_value = min(dp_hori[index_cur_row - 1][col - 1], dp_hori[index_cur_row][col - 1],
                                        dp_hori[index_cur_row + 1][col - 1])
                        if dp_hori[index_cur_row - 1][col - 1] == min_value:
                            dp_hori[index_cur_row - 1][col - 1] = invalid_value
                            index_cur_row = index_cur_row - 1
                        elif dp_hori[index_cur_row][col - 1] == min_value:
                            dp_hori[index_cur_row][col - 1] = invalid_value
                            index_cur_row = index_cur_row
                        elif dp_hori[index_cur_row + 1][col - 1] == min_value:
                            dp_hori[index_cur_row + 1][col - 1] = invalid_value
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
                num_cols_delete = 1
                cols_modify = np.zeros((rows, num_cols_delete), dtype='int32')
                cols_modify[:, :] = -1
                index_sorted_tuples = 0

                while index_cols_modify < num_cols_delete and index_sorted_tuples < cols:
                    index_cur_col = int(sorted_tuples[index_sorted_tuples][0])
                    cols_modify[rows - 1][index_cols_modify] = index_cur_col - 1
                    # index_cur_col is 1-based while col_delete is 0-based

                    for row in range(rows - 1, 0, -1):
                        min_value = min(dp_verti[row - 1][index_cur_col - 1],
                                        dp_verti[row - 1][index_cur_col],
                                        dp_verti[row - 1][index_cur_col + 1])

                        if dp_verti[row - 1][index_cur_col - 1] == min_value:
                            dp_verti[row - 1][index_cur_col - 1] = invalid_value
                            index_cur_col = index_cur_col - 1
                        elif dp_verti[row - 1][index_cur_col] == min_value:
                            dp_verti[row - 1][index_cur_col] = invalid_value
                            index_cur_col = index_cur_col
                        elif dp_verti[row - 1][index_cur_col + 1] == min_value:
                            dp_verti[row - 1][index_cur_col + 1] = invalid_value
                            index_cur_col = index_cur_col + 1

                        cols_modify[row - 1][index_cols_modify] = index_cur_col - 1
                        # index_cur_col is 1-based while col_delete is 0-based

                    if cols_modify[rows - 1][index_cols_modify] != -1:
                        index_cols_modify += 1

                    index_sorted_tuples += 1
                cost_verti = 0
                for i in range(0, index_cols_modify):
                    cost_verti += last_row[cols_modify[rows - 1][i] + 1]

            if cost_verti != -1 and cost_hori != -1:
                if cost_hori < cost_verti:
                    pic, mask = self.update_seams_hori_rm_obj(pic, rows_modify, index_rows_modify, mask)
                else:
                    pic, mask = self.update_seams_verti_rm_obj(pic, cols_modify, index_cols_modify, mask)

            elif cost_verti != -1:
                pic, mask = self.update_seams_verti_rm_obj(pic, cols_modify, index_cols_modify, mask)
            else:
                # cost_hori != -1
                pic, mask = self.update_seams_hori_rm_obj(pic, rows_modify, index_rows_modify, mask)

        cv2.imwrite(self.dst, pic)
        time_end = time.time()
        time_consumed = time_end - time_begin
        print("new picture:{}, time consumed:{}".format(self.dst, time_consumed))
        return pic
# object removal

    def maintain_obj(self, tgt_height, tgt_width):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_circle)
        self.tmp_pic = cv2.imread(self.src, 1)
        while 1:
            cv2.imshow('image', self.tmp_pic)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('a'):
                break

        self.__maintain_obj(tgt_height, tgt_width)

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            for i in range(-10, 10):
                for j in range(-10, 10):
                    self.points.append((x+i, y+j))
                    self.tmp_pic[y+j][x+i][0] = 0
                    self.tmp_pic[y+j][x+i][1] = 0
                    self.tmp_pic[y+j][x+i][2] = 0

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                for i in range(-10, 10):
                    for j in range(-10, 10):
                        self.points.append((x+i, y+j))
                        self.tmp_pic[y+j][x+i][0] = 0
                        self.tmp_pic[y+j][x+i][1] = 0
                        self.tmp_pic[y+j][x+i][2] = 0

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def __maintain_obj(self, tgt_height, tgt_width):
        points = self.points
        time_begin = time.time()
        pic = cv2.imread(self.src, 1)
        if pic is None:
            print("can't open file:{}.".format(self.src))
            return
        rows, cols = pic.shape[:2]
        diff_rows = tgt_height - rows
        diff_cols = tgt_width - cols

        invalid_value = self.get_invalid_value(pic)
        mask = np.zeros((rows, cols), dtype='int8')
        for point in points:
            x = point[0]
            y = point[1]
            mask[y][x] = 1

        while diff_rows != 0 or diff_cols != 0:
            rows, cols = pic.shape[:2]
            dp_hori = np.array([])
            dp_hori_back = np.array([])
            dp_verti = np.array([])
            dp_verti_back = np.array([])
            if diff_rows != 0:
                gradient_hori = self.laplacian(pic, True, 3) if diff_rows > 0 else self.sobel(pic, 3)
                # mask
                dp_hori, dp_hori_back = self.dp_horizontal(gradient_hori, invalid_value, mask)

            if diff_cols != 0:
                gradient_verti = self.laplacian(pic, True, 3) if diff_cols > 0 else self.sobel(pic, 3)
                # mask
                dp_verti, dp_verti_back = self.dp_vertical(gradient_verti, invalid_value, mask)

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
                rate = self.rate_insert if diff_rows > 0 else self.rate_remove
                num_rows_delete = abs(int(diff_rows * rate)) + 1
                rows_modify = np.zeros((num_rows_delete, cols), dtype='int16')
                rows_modify[:, :] = -1
                index_sorted_tuples = 0

                while index_rows_modify < num_rows_delete and index_sorted_tuples < rows:
                    index_cur_row = int(sorted_tuples[index_sorted_tuples][0])
                    rows_modify[index_rows_modify][cols - 1] = index_cur_row - 1
                    # index_cur_row is 1-based while row_delete is 0-based

                    for col in range(cols - 1, 0, -1):
                        min_value = min(dp_hori[index_cur_row - 1][col - 1], dp_hori[index_cur_row][col - 1],
                                        dp_hori[index_cur_row + 1][col - 1])
                        if min_value >= invalid_value:
                            rows_modify[index_rows_modify, :] = -1
                            break

                        if dp_hori[index_cur_row - 1][col - 1] == min_value:
                            dp_hori[index_cur_row - 1][col - 1] = invalid_value
                            index_cur_row = index_cur_row - 1
                        elif dp_hori[index_cur_row][col - 1] == min_value:
                            dp_hori[index_cur_row][col - 1] = invalid_value
                            index_cur_row = index_cur_row
                        elif dp_hori[index_cur_row + 1][col - 1] == min_value:
                            dp_hori[index_cur_row + 1][col - 1] = invalid_value
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
                rate = self.rate_insert if diff_cols > 0 else self.rate_remove
                num_cols_delete = abs(int(diff_cols * rate)) + 1
                cols_modify = np.zeros((rows, num_cols_delete), dtype='int16')
                cols_modify[:, :] = -1
                index_sorted_tuples = 0

                while index_cols_modify < num_cols_delete and index_sorted_tuples < cols:
                    index_cur_col = int(sorted_tuples[index_sorted_tuples][0])
                    cols_modify[rows - 1][index_cols_modify] = index_cur_col - 1
                    # index_cur_col is 1-based while col_delete is 0-based

                    for row in range(rows - 1, 0, -1):
                        min_value = min(dp_verti[row - 1][index_cur_col - 1],
                                        dp_verti[row - 1][index_cur_col],
                                        dp_verti[row - 1][index_cur_col + 1])
                        if min_value >= invalid_value:
                            cols_modify[:, index_cols_modify] = -1
                            break

                        if dp_verti[row - 1][index_cur_col - 1] == min_value:
                            dp_verti[row - 1][index_cur_col - 1] = invalid_value
                            index_cur_col = index_cur_col - 1
                        elif dp_verti[row - 1][index_cur_col] == min_value:
                            dp_verti[row - 1][index_cur_col] = invalid_value
                            index_cur_col = index_cur_col
                        elif dp_verti[row - 1][index_cur_col + 1] == min_value:
                            dp_verti[row - 1][index_cur_col + 1] = invalid_value
                            index_cur_col = index_cur_col + 1

                        cols_modify[row - 1][index_cols_modify] = index_cur_col - 1
                        # index_cur_col is 1-based while col_delete is 0-based

                    if cols_modify[rows - 1][index_cols_modify] != -1:
                        index_cols_modify += 1

                    index_sorted_tuples += 1
                cost_verti = 0
                for i in range(0, index_cols_modify):
                    cost_verti += last_row[cols_modify[rows - 1][i] + 1]

            if cost_verti != -1 and cost_hori != -1:
                if cost_hori < cost_verti:
                    pic, diff_rows, mask = self.update_seams_hori(pic, rows_modify, index_rows_modify, diff_rows, mask)
                else:
                    pic, diff_cols, mask = self.update_seams_verti(pic, cols_modify, index_cols_modify, diff_cols, mask)

            elif cost_verti != -1:
                pic, diff_cols, mask = self.update_seams_verti(pic, cols_modify, index_cols_modify, diff_cols, mask)
            else:
                # cost_hori != -1
                pic, diff_rows, mask = self.update_seams_hori(pic, rows_modify, index_rows_modify, diff_rows, mask)

        cv2.imwrite(self.dst, pic)
        time_end = time.time()
        time_consumed = time_end - time_begin
        print("new picture:{}, size:{}x{}, time consumed:{}".format(self.dst, tgt_height, tgt_width, time_consumed))
        return pic

    def carve(self, tgt_height, tgt_width):
        """
        :param tgt_height:target height
        :param tgt_width:target width
        :return:
        """
        time_begin = time.time()
        pic = cv2.imread(self.src, 1)
        if pic is None:
            print("can't open file:{}.".format(self.src))
            return
        rows, cols = pic.shape[:2]
        diff_rows = tgt_height - rows
        diff_cols = tgt_width - cols

        invalid_value = self.get_invalid_value(pic)

        while diff_rows != 0 or diff_cols != 0:
            rows, cols = pic.shape[:2]
            dp_hori = np.array([])
            dp_hori_back = np.array([])
            dp_verti = np.array([])
            dp_verti_back = np.array([])
            if diff_rows != 0:
                gradient_hori = self.laplacian(pic, True, 3) if diff_rows > 0 else self.sobel(pic, 3)
                dp_hori, dp_hori_back = self.dp_horizontal(gradient_hori, invalid_value)

            if diff_cols != 0:
                gradient_verti = self.laplacian(pic, True, 3) if diff_cols > 0 else self.sobel(pic, 3)
                dp_verti, dp_verti_back = self.dp_vertical(gradient_verti, invalid_value)

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
                rate = self.rate_insert if diff_rows > 0 else self.rate_remove
                num_rows_delete = abs(int(diff_rows * rate)) + 1
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
                rate = self.rate_insert if diff_cols > 0 else self.rate_remove
                num_cols_delete = abs(int(diff_cols * rate)) + 1
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

                        cols_modify[row - 1][index_cols_modify] = index_cur_col - 1
                        # index_cur_col is 1-based while col_delete is 0-based

                    if cols_modify[rows - 1][index_cols_modify] != -1:
                        index_cols_modify += 1

                    index_sorted_tuples += 1
                cost_verti = 0
                for i in range(0, index_cols_modify):
                    cost_verti += last_row[cols_modify[rows - 1][i] + 1]

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
    def update_seams_hori(pic, rows_modify, index_rows_delete, diff_rows, mask=None):
        rows, cols = pic.shape[:2]
        if diff_rows < 0:
            # remove seams
            diff_new = diff_rows + index_rows_delete
            new_pic = np.zeros((rows - index_rows_delete, cols, 3), dtype='uint8')
            new_mask = None if mask is None else np.zeros((rows - index_rows_delete, cols), dtype='uint8')
            for col in range(0, cols):
                count = 0
                for row in range(0, rows):
                    if row in rows_modify[:, col]:
                        count += 1
                    else:
                        new_pic[row - count][col] = pic[row][col]
                        if new_mask is not None:
                            new_mask[row - count][col] = mask[row][col]
        else:
            # insert seams
            diff_new = diff_rows - index_rows_delete
            new_pic = np.zeros((rows + index_rows_delete, cols, 3), dtype='uint8')
            new_mask = None if mask is None else np.zeros((rows+index_rows_delete, cols), dtype='uint8')
            for col in range(0, cols):
                count = 0
                for row in range(0, rows):
                    new_pic[row + count][col] = pic[row][col]

                    if new_mask is not None:
                        new_mask[row + count][col] = mask[row][col]

                    if row in rows_modify[:, col]:
                        count += 1

                        if new_mask is not None:
                            new_mask[row + count][col] = mask[row][col]

                        new_pic[row + count][col] = pic[row][col]
        if mask is None:
            return new_pic, diff_new
        else:
            return new_pic, diff_new, new_mask

    @staticmethod
    def update_seams_verti(pic, col_delete, index_cols_delete, diff_cols, mask=None):
        rows, cols = pic.shape[:2]
        if diff_cols < 0:
            # remove seams
            diff_new = diff_cols + index_cols_delete
            new_pic = np.zeros((rows, cols - index_cols_delete, 3), dtype='uint8')
            new_mask = None if mask is None else np.zeros((rows, cols - index_cols_delete), dtype='uint8')
            for row in range(0, rows):
                count = 0
                for col in range(0, cols):
                    if col in col_delete[row]:
                        count += 1
                    else:
                        if new_mask is not None:
                            new_mask[row][col - count] = mask[row][col]
                        new_pic[row][col - count] = pic[row][col]
        else:
            # insert seams
            diff_new = diff_cols - index_cols_delete
            new_pic = np.zeros((rows, cols + index_cols_delete, 3), dtype='uint8')
            new_mask = None if mask is None else np.zeros((rows, cols + index_cols_delete), dtype='uint8')
            for row in range(0, rows):
                count = 0
                for col in range(0, cols):
                    new_pic[row][col + count] = pic[row][col]
                    if new_mask is not None:
                        new_mask[row][col + count] = mask[row][col]
                    if col in col_delete[row]:
                        count += 1

                        if new_mask is not None:
                            new_mask[row][col + count] = mask[row][col]
                        new_pic[row][col + count] = pic[row][col]
        if mask is None:
            return new_pic, diff_new
        else:
            return new_pic, diff_new, new_mask

    @staticmethod
    def update_seams_hori_rm_obj(pic, rows_modify, index_rows_delete, mask):
        rows, cols = pic.shape[:2]
        new_pic = np.zeros((rows - index_rows_delete, cols, 3), dtype='uint8')
        new_mask = np.zeros((rows - index_rows_delete, cols), dtype='uint8')
        for col in range(0, cols):
            count = 0
            for row in range(0, rows):
                if row in rows_modify[:, col]:
                    count += 1
                else:
                    new_pic[row - count][col] = pic[row][col]
                    new_mask[row - count][col] = mask[row][col]
        return new_pic, new_mask

    @staticmethod
    def update_seams_verti_rm_obj(pic, col_delete, index_cols_delete, mask):
        rows, cols = pic.shape[:2]
        new_pic = np.zeros((rows, cols - index_cols_delete, 3), dtype='uint8')
        new_mask = np.zeros((rows, cols - index_cols_delete), dtype='uint8')
        for row in range(0, rows):
            count = 0
            for col in range(0, cols):
                if col in col_delete[row]:
                    count += 1
                else:
                    if new_mask is not None:
                        new_mask[row][col - count] = mask[row][col]
                    new_pic[row][col - count] = pic[row][col]
        return new_pic, new_mask


    @staticmethod
    def dp_horizontal_rm_obj(pic, invalid_value, mask):
        # use dp to calculate the energy for each row
        rows, cols = pic.shape[:2]
        dp = np.zeros((rows + 2, cols), dtype='int32')
        dp[1:rows + 1, 0:cols] = pic[0:rows, 0:cols]
        dp_back = np.zeros((rows + 2, cols), dtype='int32')
        dp_back[1:rows + 1, 0:cols] = pic[0:rows, 0:cols]

        for row in range(rows):
            for col in range(cols):
                if mask[row][col] == 1:
                    dp[row + 1][col] = -invalid_value
                    dp_back[row + 1][col] = -invalid_value

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
    def dp_vertical_rm_obj(pic, invalid_value, mask):
        # use dp to calculate the energy for each column
        rows, cols = pic.shape[:2]
        dp = np.zeros((rows, cols + 2), dtype='int32')
        dp[0:rows, 1:cols + 1] = pic[0:rows, 0:cols]
        dp_back = np.zeros((rows, cols + 2), dtype='int32')
        dp_back[0:rows, 1:cols + 1] = pic[0:rows, 0:cols]

        for row in range(rows):
            for col in range(cols):
                if mask[row][col] == 1:
                    dp[row][col + 1] = -invalid_value
                    dp_back[row][col + 1] = -invalid_value

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
    def dp_horizontal(pic, invalid_value, mask=None):
        # use dp to calculate the energy for each row
        rows, cols = pic.shape[:2]
        dp = np.zeros((rows + 2, cols), dtype='int32')
        dp[1:rows + 1, 0:cols] = pic[0:rows, 0:cols]
        dp_back = np.zeros((rows + 2, cols), dtype='int32')
        dp_back[1:rows + 1, 0:cols] = pic[0:rows, 0:cols]

        if mask is not None:
            for row in range(rows):
                for col in range(cols):
                    if mask[row][col] == 1:
                        dp[row + 1][col] = invalid_value
                        dp_back[row + 1][col] = invalid_value

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
    def dp_vertical(pic, invalid_value, mask = None):
        # use dp to calculate the energy for each column
        rows, cols = pic.shape[:2]
        dp = np.zeros((rows, cols + 2), dtype='int32')
        dp[0:rows, 1:cols + 1] = pic[0:rows, 0:cols]
        dp_back = np.zeros((rows, cols + 2), dtype='int32')
        dp_back[0:rows, 1:cols + 1] = pic[0:rows, 0:cols]

        if mask is not None:
            for row in range(rows):
                for col in range(cols):
                    if mask[row][col] == 1:
                        dp[row][col + 1] = invalid_value
                        dp_back[row][col + 1] = invalid_value

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
