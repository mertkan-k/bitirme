import sys, os
from copy import copy

import cv2
cv = cv2

import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import statistics
import glob

from random import randint
from pprint import pprint

np.set_printoptions(threshold=sys.maxsize)

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def Avg(l):
	return statistics.mean(l)

def IsNear(val, l):
	# print(val, l)
	if len(l) == 0:
		return True
	else:
		return abs(Avg(l) - val) < 8

def GetRandomColor():
	return (randint(0,256), randint(0,256), randint(0,256))

def EnsureDir(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)

def MultiplySize(w, h, multiplier):
	return w * multiplier, h * multiplier

def Error(str):
	print("Error: {}".format(str))

def Log(str, level = 1):
	WARNING_LEVEL = 1
	if level >= WARNING_LEVEL:
		print("Warning: {}".format(str))

class Rect():
	@staticmethod
	def GetTextOfImg(img, lang='tur'):
		text = pytesseract.image_to_string(img, lang=lang, config='--oem 3 --psm 6').strip()
		# text = pytesseract.image_to_string(img, lang=lang, config="-c tessedit_char_blacklist='?.' --psm 6").strip()
		return text

	def __init__(self, x, y, w, h):
		self.x = x
		self.y = y
		self.w = w
		self.h = h

		self.color = (0, 255, 0)

	def GetPosition(self):
		return (self.x, self.y)
	def GetStartPoint(self):
		return self.GetPosition()
	def GetSize(self):
		return (self.w, self.h)
	def GetEndPoint(self): ## right bottom
		return tuple(map(sum, zip(self.GetPosition(), self.GetSize())))
	def GetArea(self):
		return self.w * self.h

	def GetCommonArea(self, other_rect):
		up, down = None, None
		if self.y < other_rect.y:
			up = self
			down = other_rect
		else:
			up = other_rect
			down = self

		x = int(up.x + up.w/4)
		y = int(down.y + down.h/4)
		w = int(up.w/2)
		h = int(down.h/2)

		return (x, y, w, h)

	def GetText(self, img):
		TEXT_ADD = 0
		text_img = img[self.y:self.y + self.h, self.x:self.x + self.w]
		self.text = self.GetTextOfImg(text_img)
		return self.text

	def Draw(self, img):
		cv.rectangle(img, self.GetStartPoint(), self.GetEndPoint(), self.color, 1)

	def __repr__(self):
		return "Rect(Start{}, End{})".format(self.GetPosition(), self.GetEndPoint())

class ImageTester():

	class Template():
		def __init__(self, name, img_path, color):
			self.name = name
			self.img_path = img_path
			self.color = color

			self.img = cv.imread(img_path, 0)

	def __init__(self):
		self.image_path = None

		self.img_rgb = None
		self.img_gray = None
		self.img_out = None
		self.img_blur = None

		self.templates = []
		self.founded_contours = []

	def SetImageName(self, name):
		self.image_path = name

		self.GeneratePaths()
		self.GenerateImages()

	def GenerateImages(self):
		THRESHOLDVALUE = 180

		self.img_rgb = cv.imread(self.image_path)
		self.img_out = copy(self.img_rgb)
		self.img_gray = cv.cvtColor(self.img_rgb, cv.COLOR_BGR2GRAY)
		self.img_blur = cv2.medianBlur(self.img_gray, 3)
		_, self.img_thres = cv.threshold(self.img_gray, THRESHOLDVALUE, 255, 0)

	def GeneratePaths(self):
		if os.path.isfile(self.image_path) == False:
			Error("test image not found '{}'".format(self.image_path))
			return False

	def Show(self, img, cmap = 'viridis'):
		plt.grid(color = 'purple', linestyle = '--', linewidth = 0.5, alpha = 0.5)
		plt.imshow(img, cmap = cmap)
		plt.show()

	def AddTemplate(self, name, templatePath, color = (255, 255, 255)):
		if not os.path.isfile(templatePath):
			print("No image {}".format(templatePath))
			return False

		self.templates.append(self.Template(name, templatePath, color))

		return self.templates[-1]

	def FindLines(self):
		ITERATIONS = 3

		img_bin = 255 - self.img_thres

		kernel_len = np.array(self.img_gray).shape[1]//100
		ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
		hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

		image_1 = cv2.erode(img_bin, ver_kernel, iterations=ITERATIONS)
		self.img_lines_vertical = cv2.dilate(image_1, ver_kernel, iterations=ITERATIONS)
		# cv.imshow('Vertical Test', self.img_lines_vertical)

		image_2 = cv2.erode(img_bin, hor_kernel, iterations=ITERATIONS)
		self.img_lines_horizontal = cv2.dilate(image_2, hor_kernel, iterations=ITERATIONS)
		# cv.imshow('Horizontal Test', self.img_lines_horizontal)

		img_vh = cv2.addWeighted(self.img_lines_vertical, 0.5, self.img_lines_horizontal, 0.5, 0.0)
		img_vh = cv2.erode(~img_vh, kernel, iterations=2)
		thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
		# bitxor = cv2.bitwise_xor(self.img_gray, img_vh)
		# bitnot = cv2.bitwise_not(bitxor)

		self.img_lines = img_vh
		# self.Show(self.img_lines)

	def FindContours(self, threshold = 200):
		self.img_contours = copy(self.img_rgb)

		self.contours, hierarchy = cv.findContours(self.img_lines, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		self.rectangles = []

		for contour in self.contours:
			(x,y,w,h) = cv2.boundingRect(contour)
			if w < 10 or h < 10:
				continue
			self.rectangles.append(Rect(x,y,w,h))
			cv2.rectangle(self.img_contours, (x,y), (x+w,y+h), GetRandomColor(), 1)

		# self.Show(self.img_contours)

	def DedectRowsAndCols(self):
		self.DedectCols()
		self.SortCols()
		self.SortRowsInCols()

	def DedectCols(self):
		rectangles = copy(self.rectangles)[2::]
		allCols = []

		for rect in rectangles:
			isFounded = False

			for eachRow in allCols:
				if IsNear(rect.x, [ r.x for r in eachRow ]):
					eachRow.append(rect)
					isFounded = True
					break

			if not isFounded:
				allCols.append([rect])

		self.Cols = allCols

	def SortCols(self):
		avgXOfCols = [] ## [col_index, avgX]

		for colIndex in range(len(self.Cols)):
			avgXOfCols.append([colIndex, Avg([ r.x for r in self.Cols[colIndex] ])])

		avgXOfCols = sorted(avgXOfCols, key=lambda lst: lst[1])
		newCols = []

		for index, avg in avgXOfCols:
			newCols.append(self.Cols[index])

		self.Cols = newCols

	def SortRowsInCols(self):
		for colIndex in range(len(self.Cols)):
			self.Cols[colIndex] = sorted(self.Cols[colIndex], key=lambda rect: rect.y)

	def SplitCols(self):
		cols = self.Cols

		title = cols[0][0]

		cols[0].pop(0) ## for title
		cols[0].pop(0) ## for empty blank

		questions = [ r for r in cols[0] ]
		answers = [ cols[colIndex+1][0] for colIndex in range(len(cols)-1) ]

		self.data = (title, questions, answers)

	def ShowTest(self):
		title, questions, answers = self.data

		test_img = copy(self.img_rgb)
		title.color = (255, 0, 0)
		title.Draw(test_img)

		for q in questions:
			q.color = (0, 255, 0)
			q.Draw(test_img)

		for a in answers:
			a.color = (0, 0, 255)
			a.Draw(test_img)

		self.Show(test_img)

	def PrintTests(self):
		title, questions, answers = self.data
		text_img = self.img_thres

		title = title.GetText(text_img)

		questions = [ q.GetText(text_img) for q in questions ]
		answers = [ a.GetText(text_img) for a in answers ]

		print('Title : {}'.format(title))
		print('Questions : {}'.format(questions))
		print('Answers : {}'.format(answers))

	def Process(self):
		self.FindLines()
		self.FindContours()
		self.DedectRowsAndCols()
		self.SplitCols()

		# self.ShowTest()
		# self.PrintTests()

class SurveyTester():
	TEST_PATH = r"tests/"

	class Question(Rect):
		@staticmethod
		def GetMergeFormat(workbook):
			return workbook.add_format({
			# 'bold':     True,
			# 'border':   6,
			'align':    'left',
			'valign':   'vcenter',
			# 'fg_color': '#D7E4BC',
		})

	class Answer(Rect):
		@staticmethod
		def GetMergeFormat(workbook):
			return workbook.add_format({
			# 'bold':     True,
			# 'border':   6,
			'align':    'center',
			'valign':   'vcenter',
			# 'fg_color': '#D7E4BC',
		})

	class Title(Rect):
		@staticmethod
		def GetMergeFormat(workbook):
			return workbook.add_format({
			# 'bold':     True,
			# 'border':   6,
			'align':    'center',
			'valign':   'vcenter',
			'fg_color': '#D7E4BC',
		})

	def __init__(self, surveyDir):
		super().__init__()

		self.title = None
		self.questions = []
		self.answers = []

		self.excelWB = None
		self.excelWS = None

		surveyDir = os.path.join(self.TEST_PATH, surveyDir)
		self.surveyDir = surveyDir

		self.Init()

	def Init(self):
		if not os.path.exists(self.surveyDir):
			print("Survey path does not exist {}", self.surveyDir)
			sys.exit(1)

		EXTENSIONS = ['.png', '.jpg']

		self.surveyPaths = []
		for f in os.listdir(self.surveyDir):
			ext = os.path.splitext(f)[1]
			if ext.lower() not in EXTENSIONS:
				continue
			self.surveyPaths.append(f)

		if len(self.surveyPaths) == 0:
			print("Survey path does not have any file {}".format(self.surveyDir))
			sys.exit(1)

		Log("Survey file count {}".format(len(self.surveyPaths)))
		# Log("self.surveyPaths {}".format(self.surveyPaths))

		self.surveyName = os.path.basename(self.surveyDir)
		self.resultDir = os.path.join(self.surveyDir, "results/")
		# EnsureDir(self.resultDir)

		# self.excelWB = xlsxwriter.Workbook(os.path.join(self.resultDir, "result.xlsx"))
		# self.excelWS = self.excelWB.add_worksheet()

	def CreateFinalWS(self, qCount, aCount):
		worksheets = self.excelWB.worksheets()
		if len(worksheets) == 0:
			return

		worksheet = self.excelWB.add_worksheet('Results')

		for row in range(qCount):
			worksheet.write(row+1, 0, "Q-%d" % (row+1))
		# worksheet.write(row+2, 0, "T")

		for col in range(aCount):
			worksheet.write(0, col+1, "A-%d" % (col+1))
		# worksheet.write(0, col+2, "T")

		xl_rowcol_to_cell = xlsxwriter.utility.xl_rowcol_to_cell

		firstSheet = worksheets[0].get_name()
		lastSheet = worksheets[-2].get_name()

		for row in range(qCount):
			for col in range(aCount):
				cell = xl_rowcol_to_cell(row+1+1, col+1)
				worksheet.write(row+1, col+1, '=SUM(%s:%s!%s)' % (firstSheet, lastSheet, cell))

		worksheets.insert(0, worksheets.pop()) ## move results to first sheet

	def Analyze(self, surveyPath, img, data):
		title, questions, answers = data

		worksheet = self.excelWB.add_worksheet(surveyPath)

		row = 1+1
		for q in questions:
			worksheet.write(row, 0, q.GetText(img))
			row += 1

		col = 1
		for a in answers:
			worksheet.write(1, col, a.GetText(img))
			col += 1

		if title == None:
			Error("title not found!")
			worksheet.merge_range(0, 0, 0, len(answers), "Not Titled", self.Title.GetMergeFormat(self.excelWB))
		else:
			worksheet.merge_range(0, 0, 0, len(answers), title.GetText(img), self.Title.GetMergeFormat(self.excelWB))

		# Log("Total questions: {}".format(len(questions)))
		# Log("Total answers: {}".format(len(answers)))

		row = 1+1
		for q in questions:
			col = 1

			for a in answers:
				x, y, w, h =  q.GetCommonArea(a)
				# Log("Common Area {} {} {}".format(q.GetText(), a.GetText(), (x, y, w, h)))

				common_area_image = img[y:y+h, x:x+w]
				number_of_black_pix = np.sum(common_area_image == 0)

				if number_of_black_pix > 25:
					worksheet.write_number(row, col, 1)
				else:
					worksheet.write_number(row, col, 0)
				# worksheet.write_number(row, col, number_of_black_pix)
				col += 1

			row += 1

		return len(questions), len(answers)

	def Process(self):
		self.excelWB = xlsxwriter.Workbook(os.path.join(self.surveyDir, "result.xlsx"))

		qCount, aCount = 0, 0

		# self.surveyPaths = ['1_clear.png']

		for surveyPath in self.surveyPaths:
			Log("{} processing..".format(surveyPath))
			it = ImageTester()
			it.SetImageName(os.path.join(self.surveyDir, surveyPath))
			it.Process()

			qC, aC = self.Analyze(surveyPath, it.img_thres, it.data)
			qCount, aCount = max(qC, qCount), max(aC, aCount)

		self.CreateFinalWS(qCount, aCount)

		self.excelWB.close()

		os.startfile(os.path.join(self.surveyDir, "result.xlsx"))

if __name__ == "__main__":
	s1 = SurveyTester("paint_test_1")
	s1.Process()





