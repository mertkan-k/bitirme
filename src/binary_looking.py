import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
try:
	from PIL import Image
except ImportError:
	import Image
import pytesseract

#read your file
image_index = 2
file =  r'tests/test_images/%d.png' % image_index
img = cv2.imread(file,0)
# img.shape

#thresholding the image to a binary image
thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)

#inverting the image
img_bin = 255-img_bin
cv2.imwrite("tests/results/%d/detected.jpg" % image_index, img_bin)

#Plotting the image to see the output
plotting = plt.imshow(img_bin, cmap='gray')
plt.show()