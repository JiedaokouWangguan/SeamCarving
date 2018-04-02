from seamCarver.seam_carver import SeamCarver

sc = SeamCarver("pic.png", "modified_pic.png", 0.27, 0.65)
rows, cols = sc.get_rows_cols()
# pic = sc.carve(rows, cols - 200)
# sc.remove_obj(False, True)
pic = sc.maintain_obj(rows, cols - 350)
