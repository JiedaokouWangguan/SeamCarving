import cv2
import numpy as np

class SeamCarver(Object):


	def __init__(self, src, dst):
		self.src = src
		self.dst = dst
		self.pic = cv2.imread(src, 1)
		self.pic_modified = []

	def carve(self, pic, tgt_height, tgt_width):
		# 修改图片,返回修改后的图片
		pass


	def dp(self, pic, hori, verti):
		# 动态规划算出来每行, 或者每列的energy
		pass


	def laplacian(self, pic, blur, gaussian_ksize):
		# 先高斯模糊，再拉普拉斯
        gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        if blur:
        	gray = cv2.GaussianBlur(gray, (gaussian_ksize, gaussian_ksize), 0)
        grad = cv2.Laplacian(gray, cv2.CV_64F)
        grad = cv2.convertScaleAbs(grad, grad)
		return grad


	def sobel(self, pic, ksize):
		# 使用sobel算子
		gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=ksize)
        grad = cv2.convertScaleAbs(grad, grad)
		return grad