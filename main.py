from seamCarving.seam_carver import SeamCarver

sc = SeamCarver("pic.png", "modified_pic.png", 0.5)
rows, cols = sc.get_rows_cols()
pic = sc.carve(rows+100, cols-100)  # all the argument errors
