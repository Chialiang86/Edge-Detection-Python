import cv2
import numpy as np
from matplotlib import pyplot as plt


def hist_xy(img):
	x = range(256)
	y = np.zeros(256)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			y[img[i][j]] += 1
	return x, y

def histshow(img1, img2, fname):
	x1, y1 = hist_xy(img1)
	x2, y2 = hist_xy(img2)
	plt.subplot(2, 1, 1)
	plt.plot(x1, y1)
	plt.title('before')
	plt.subplot(2, 1, 2)
	plt.plot(x2, y2)
	plt.title('after')
	plt.savefig(fname)
	plt.show()
	

imgnum = 2
for i in range(imgnum):
	img = cv2.imread('src/' + str(i) + ".jpg")

	# img to gray
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# histagram equilization
	eq_img = cv2.equalizeHist(gray_img)
	histshow(gray_img, eq_img,str(i) + '/Histogram_equalization' + '.jpg')

	# Gaussian blurred
	res = np.hstack([gray_img, eq_img])
	gray_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
	eq_img = cv2.GaussianBlur(eq_img, (7, 7), 0)

	# show Gaussian blurred res
	cv2.imwrite(str(i) + '/Gaussian_blurred' + '.jpg', eq_img)

	# show edge detecttion

	# use gray image 
	# 對 x 微分、對 y 微分、加權平均 依序串接成一張
	gray_img_edgex = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0)
	gray_img_edgey = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1)
	gray_img_edgeabsx = cv2.convertScaleAbs(gray_img_edgex)
	gray_img_edgeabsy = cv2.convertScaleAbs(gray_img_edgey)
	gray_img_edge = cv2.addWeighted(gray_img_edgeabsx, 0.5, gray_img_edgeabsy, 0.5, 0)
	gray_edge = np.hstack((gray_img_edgeabsx, gray_img_edgeabsy, gray_img_edge))
	cv2.imwrite(str(i) + '/Edge_gray' + '.jpg', gray_edge)

	# use equalization image 
	# 對 x 微分、對 y 微分、加權平均 依序串接成一張
	eq_img_edgex = cv2.Sobel(eq_img, cv2.CV_16S, 1, 0)
	eq_img_edgey = cv2.Sobel(eq_img, cv2.CV_16S, 0, 1)
	eq_img_edgeabsx = cv2.convertScaleAbs(eq_img_edgex)
	eq_img_edgeabsy = cv2.convertScaleAbs(eq_img_edgey)
	eq_img_edge = cv2.addWeighted(eq_img_edgeabsx, 0.5, eq_img_edgeabsy, 0.5, 0)
	eq_edge = np.hstack((eq_img_edgeabsx, eq_img_edgeabsy, eq_img_edge))
	cv2.imwrite(str(i) + '/Edge_equalization' + '.jpg', eq_edge)

cv2.waitKey(0)
cv2.destroyAllWindows()

	
