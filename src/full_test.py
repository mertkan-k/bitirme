import sys
from random import randint

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

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

SHOP_SLOT = 1

image_index = 11
file =  r'tests/test_images/%d.png' % image_index
res_path = "tests/results/%d/" % image_index

img = cv2.imread(file,0)
img.shape

#thresholding the image to a binary image
thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#inverting the image
img_bin = 255-img_bin
cv2.imwrite(res_path + 'cv_inverted.png',img_bin)
#Plotting the image to see the output
plotting = plt.imshow(img_bin,cmap='gray')
print(1)
if SHOP_SLOT: plt.show()

# countcol(width) of kernel as 100th of total width
kernel_len = np.array(img).shape[1]//100
# Defining a vertical kernel to detect all vertical lines of image
ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
# Defining a horizontal kernel to detect all horizontal lines of image
hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
# A kernel of 2x2
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

#Use vertical kernel to detect and save the vertical lines in a jpg
image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
cv2.imwrite(res_path + "vertical.jpg",vertical_lines)
#Plot the generated image
plotting = plt.imshow(image_1,cmap='gray')
print(2)
if SHOP_SLOT: plt.show()

#Use horizontal kernel to detect and save the horizontal lines in a jpg
image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
cv2.imwrite(res_path + "horizontal.jpg",horizontal_lines)
#Plot the generated image
plotting = plt.imshow(image_2,cmap='gray')
print(3)
if SHOP_SLOT: plt.show()

# Combine horizontal and vertical lines in a new third image, with both having same weight.
img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
#Eroding and thesholding the image
img_vh = cv2.erode(~img_vh, kernel, iterations=2)
thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite(res_path + "img_vh.jpg", img_vh)
bitxor = cv2.bitwise_xor(img,img_vh)
bitnot = cv2.bitwise_not(bitxor)
#Plotting the generated image
plotting = plt.imshow(bitnot,cmap='gray')
print(4)
if SHOP_SLOT: plt.show()

# Detect contours for following box detection
contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
	key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

# Sort all the contours by top to bottom.
contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

#Creating a list of heights for all detected boxes
heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

#Get mean of heights
mean = np.mean(heights)

def GetTextOfImg(img):
	# , config='--psm 3'
	text = pytesseract.image_to_string(img, lang='tur').strip()
	return text

#Create list box to store all boxes in
box = []
# Get position (x,y), width and height for every contour and show the contour on image
dedect_index = 0
for c in contours:
	x, y, w, h = cv2.boundingRect(c)
	if (w<1000 and h<500):
		# color = (randint(10, 240), randint(10, 240), randint(10, 240))
		color = (0, 255, 0)
		image = cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
		box.append([x,y,w,h])

		cropped = img[y:y + h, x:x + w]

		text = GetTextOfImg(cropped)

		if text != "" and len(text) >= 1:
			print(dedect_index, " >>", x, y, w, h, len(text), '{', text, '}')
			print('-' * 90)

			cv2.putText(img, str(dedect_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

			dedect_index += 1


plotting = plt.imshow(image,cmap='gray')


print(5)
plt.show()

#Creating two lists to define row and column in which cell is located
row=[]
column=[]
j=0

#Sorting the boxes to their respective row and column
for i in range(len(box)):

	if(i==0):
		column.append(box[i])
		previous=box[i]

	else:
		if(box[i][1]<=previous[1]+mean/2):
			column.append(box[i])
			previous=box[i]

			if(i==len(box)-1):
				row.append(column)

		else:
			row.append(column)
			column=[]
			previous = box[i]
			column.append(box[i])

print(column)
print(row)

#calculating maximum number of cells
countcol = 0
for i in range(len(row)):
	countcol = len(row[i])
	if countcol > countcol:
		countcol = countcol

#Retrieving the center of each column
center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]

center=np.array(center)
center.sort()
print(center)
#Regarding the distance to the columns center, the boxes are arranged in respective order

finalboxes = []
for i in range(len(row)):
	lis=[]
	for k in range(countcol):
		lis.append([])
	for j in range(len(row[i])):
		diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
		minimum = min(diff)
		indexing = list(diff).index(minimum)
		lis[indexing].append(row[i][j])
	finalboxes.append(lis)


#from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
outer=[]
for i in range(len(finalboxes)):
	for j in range(len(finalboxes[i])):
		inner=''
		if(len(finalboxes[i][j])==0):
			outer.append(' ')
		else:
			for k in range(len(finalboxes[i][j])):
				y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
				finalimg = bitnot[x:x+h, y:y+w]
				kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
				border = cv2.copyMakeBorder(finalimg,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
				resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
				dilation = cv2.dilate(resizing, kernel,iterations=1)
				erosion = cv2.erode(dilation, kernel,iterations=2)

				out = pytesseract.image_to_string(erosion, lang='tur')
				if(len(out)==0):
					out = pytesseract.image_to_string(erosion, lang='tur', config='--psm 3')
				inner = inner +" "+ out
			outer.append(inner)

#Creating a dataframe of the generated OCR list
arr = np.array(outer)
dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
print(dataframe)
data = dataframe.style.set_properties(align="left")
#Converting it in a excel-file
data.to_excel(res_path + "output.xlsx", engine='xlsxwriter')