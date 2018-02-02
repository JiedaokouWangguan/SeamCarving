from seamCarver.seam_carver import SeamCarver

sc = SeamCarver("painting.jpg", "modified_painting.jpg", 0.6, 0.4)
rows, cols = sc.get_rows_cols()
pic = sc.carve(rows+50, cols + 300)
