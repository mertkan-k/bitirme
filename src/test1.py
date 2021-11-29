import sys, os

import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

from random import randint

# from pprint import pprint as print

def ensure_dir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

def GetTextOfImg(img):
	text = pytesseract.image_to_string(img, lang='tur').strip()
	return text

image_index = 11

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# file =  r'C:\Users\mertk\Desktop\lessons\bitirme\src\1.jpg'
file =  r'tests/test_images/%d.png' % image_index

im1 = cv2.imread(file, 0)
im = cv2.imread(file)

# im1 = cv2.imread(file, cv2.imread.CV_LOAD_IMAGE_COLOR)
# im = cv2.cv.LoadImage(file, CV_LOAD_IMAGE_COLOR)

# print(file)
# print(im1)
# print(im)

ret, thresh_value = cv2.threshold(im1,180,255,cv2.THRESH_BINARY_INV)

kernel = np.ones((5, 5), np.uint8)
dilated_value = cv2.dilate(thresh_value, kernel, iterations = 1)

contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cordinates = []

ensure_dir("tests/results/%d/" % image_index)

file = open("tests/results/%d/text.txt" % image_index, "w", encoding="utf-8")
original_stdout = sys.stdout
sys.stdout = file

dedect_index = 0

for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	# print(x,y,w,h)
	cordinates.append((x,y,w,h))
	#bounding the images
	if y < 50:
		pass

	# color = (0, 0, 255)

	cropped = im1[y:y + h, x:x + w]

	# Apply OCR on the cropped image
	# print(pytesseract.get_languages())
	text = GetTextOfImg(cropped)

	if text != "" and len(text) >= 3:
		print(dedect_index, " >>", x, y, w, h, len(text), '{', text, '}')
		print('-' * 90)

		# print(x,y,w,h,len(text),text, )
		# print('-'*100)

		# print(len(text))
		# print(len(text), text, ord(text))

		color = (randint(10, 240), randint(10, 240), randint(10, 240))
		cv2.rectangle(im, (x, y), (x+w,y+h), color, 1)
		# cv2.putText(im, 'Hello World!', (x, y))
		cv2.putText(im, str(dedect_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

		dedect_index +=1

plt.imshow(im)
cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)
cv2.imwrite("tests/results/%d/detected.jpg" % image_index, im)

file.close()

# a = input('sad')
