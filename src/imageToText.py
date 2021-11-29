import sys, os
from copy import copy

import cv2 as cv
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

from random import randint
from pprint import pprint

np.set_printoptions(threshold=sys.maxsize)

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def EnsureDir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

def Error(str):
	print("Error: {}".format(str))

def Warning(str, level = 1):
	WARNING_LEVEL = 1
	if level >= WARNING_LEVEL:
		print("Warning: {}".format(str))

class ImageTester():
	GENERAL_PATH = r'tests'
	IMAGE_PATH = os.path.join(GENERAL_PATH, 'images/')
	RESULT_PATH = os.path.join(GENERAL_PATH, 'results/')

	TRIANGLE_PATH = os.path.join(GENERAL_PATH, 'sign', 'ucgen_real.png')
	SQUARE_PATH = os.path.join(GENERAL_PATH, 'sign', 'ucgen.png')
	CIRCLE_PATH = os.path.join(GENERAL_PATH, 'sign', 'yuvarlak.png')

	@staticmethod
	def GetTextOfImg(img, lang='tur'):
		text = pytesseract.image_to_string(img, lang=lang).strip()
		return text

	@staticmethod
	## will be overrite in file
	## todo:
	def ProcessFunction():
		pass

	def __init__(self):
		self.image_name = None

		self.image_path = None
		self.result_path = None

		self.img_rgb = None
		self.img_out = None
		self.out_file = None

		self.Clear()

	def Clear(self):
		self.founded_objects = []
		self.founded_contours = []
		self.matched_list = []

	# def __del__(self):
	# 	self.out_file.close()

	def SetImageName(self, name):
		self.image_name = name

		self.GeneratePaths()
		self.GenerateOutFile()

	def GeneratePaths(self):
		self.image_path = os.path.join(self.IMAGE_PATH, self.image_name)
		if os.path.isfile(self.image_path) == False:
			Error("test image not found '{}'".format(self.image_path))
			return False

		self.img_rgb = cv.imread(self.image_path)
		self.img_out = copy(self.img_rgb)

		self.result_path = os.path.join(self.RESULT_PATH, self.image_name + '/')
		EnsureDir(self.result_path)

	def GenerateOutFile(self):
		self.out_file = open(self.GetSubResultPath("textout.py"), "w", encoding="utf-8")

	def GetSubResultPath(self, fileName):
		return os.path.join(self.result_path, fileName)

	def Show(self, img):
		plt.imshow(img)
		plt.show()

	def TestMultiple(self, templateSubDir):
		img_gray = cv.cvtColor(self.img_rgb, cv.COLOR_BGR2GRAY)
		template = cv.imread(os.path.join(self.GENERAL_PATH, 'sign', templateSubDir),0)
		SCALE_EACH_LOOP = 0.93
		MIN_THRESOLD = 0.8
		MIN_TEMPLATE_SIZE = (5, 5)

		while True:
			w, h = template.shape[::-1]
			if MIN_TEMPLATE_SIZE[0] > w or  MIN_TEMPLATE_SIZE[1] > h:
				break

			res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
			threshold = 1.0

			while True:
				loc = np.where( res >= threshold)
				count = 0
				for pt in zip(*loc[::-1]):
					isFounded = False
					FOUN_SENS_PIXEL_COUNT = 2
					for fx, fy, fw, fh in self.founded_objects:
						if pt[0] >= fx-FOUN_SENS_PIXEL_COUNT and pt[0] <= fx+w+FOUN_SENS_PIXEL_COUNT:
							if pt[1] >= fy-FOUN_SENS_PIXEL_COUNT and pt[1] <= fy+h+FOUN_SENS_PIXEL_COUNT:
								isFounded = True
								break
					if isFounded: break

					cv.rectangle(self.img_out, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
					self.founded_objects.append((*pt, pt[0] + w, pt[1] + h))
					print("Founded rectangle:", pt, w, h)
					count += 1
				if count == 0:
					threshold -= 0.005
					# print("Threshold log:", threshold)
				else:
					# print(count)
					print("Found threshold:", threshold, "Len:", len(self.founded_objects))
					# break
				if threshold < MIN_THRESOLD:
					break
			if len(self.founded_objects) == 0:
				width = int(w * SCALE_EACH_LOOP)
				height = int(h * SCALE_EACH_LOOP)
				dim = (width, height)
				# print("Scaled:", dim)
				template = cv.resize(template, dim, interpolation = cv.INTER_AREA)

			else:
				print("Final template size:", w, h)
				break
		print("Final rectange count:", len(self.founded_objects))

	def FindContours(self):
		imgray = cv.cvtColor(self.img_rgb, cv.COLOR_BGR2GRAY)
		ret, thresh = cv.threshold(imgray, 127, 255, 0)
		self.founded_contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		print("Founded contour count: ", len(self.founded_contours))

	def CreateContourDict(self):
		MIN_CONTOUR_AREA = 250

		for contour in self.founded_contours[::-1]:
			match_count = 0
			x, y, w, h = cv.boundingRect(contour)
			if w * h < MIN_CONTOUR_AREA:
				# print("Low contour area")
				continue

			for rl, rt, rr, rb in self.founded_objects:
				if (int(x) < rl and int(x+w) > rr):
					if (int(y) < rt and int(y+h) > rb):
						# print("Matched: ", x, y, w, h, rl, rt, rr, rb)
						match_count += 1
						cv.rectangle(self.img_out, (x, y), (x+w, y+h), (0,255,0), 1)
						self.founded_objects.remove((rl, rt, rr, rb))

			if match_count == 1:
				self.matched_list.append((x, y, w, h))
			# elif match_count > 1:
			# 	print("Multi match")
			# elif match_count < 1:
			# 	print("No match")

		print("Matched size: ", len(self.matched_list))

	def Process(self, method = None):
		self.TestMultiple('kare.png')
		self.FindContours()
		self.CreateContourDict()
		self.Clear()

		self.TestMultiple('ucgen.png')
		self.FindContours()
		self.CreateContourDict()
		self.Clear()

		self.TestMultiple('yuvarlak.png')
		self.FindContours()
		self.CreateContourDict()

		self.Show(self.img_out)

EnsureDir(ImageTester.IMAGE_PATH)
EnsureDir(ImageTester.RESULT_PATH)

it1 = ImageTester()
it1.SetImageName("excel_ornegi.png")
it1.Process()
