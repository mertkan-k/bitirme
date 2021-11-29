
		Warning("process started for '{}'".format(self.image_name))

		im1 = cv2.imread(self.image_path, 0)
		im = cv2.imread(self.image_path)

		# im1 = cv2.imread(self.image_path, cv2.imread.CV_LOAD_IMAGE_COLOR)
		# im = cv2.cv.LoadImage(self.image_path, CV_LOAD_IMAGE_COLOR)

		ret, thresh_value = cv2.threshold(im1,180,255,cv2.THRESH_BINARY_INV)

		kernel = np.ones((5, 5), np.uint8)
		dilated_value = cv2.dilate(thresh_value, kernel, iterations = 1)

		contours, hierarchy = cv2.findContours(dilated_value, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		Warning("Founded contour count '{}'".format(len(contours)))

		file = open(self.GetSubResultPath("texts.txt"), "w", encoding="utf-8")
		CHANGE_OUT = 1
		if CHANGE_OUT:
			original_stdout = sys.stdout
			sys.stdout = file

		dedect_index = 0

		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
			# print(x,y,w,h)
			#bounding the images
			if w < 10:
				Warning("Passed rect, w is so small (w: {})".format(w), 0)
				continue
			if h < 10:
				Warning("Passed rect, w is so small (h: {})".format(w), 0)
				continue

			# color = (0, 0, 255)

			cropped = im1[y:y + h, x:x + w]

			# Apply OCR on the cropped image
			# print(pytesseract.get_languages())
			text = self.GetTextOfImg(cropped)

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

		if CHANGE_OUT:
			sys.stdout = original_stdout

		plt.imshow(im)
		cv2.namedWindow('detecttable', cv2.WINDOW_NORMAL)
		cv2.imwrite(self.GetSubResultPath('detecttable.jpg'), im)

		file.close()
		Warning("process ended for '{}'".format(self.image_name))



	def TestMethod2(self):
		# Read the main image
		img_rgb = cv.imread(self.image_path)

		# Convert to grayscale
		img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

		# Read the template
		template = cv.imread(self.CIRCLE_PATH,0)

		# Store width and height of template in w and h
		w, h = template.shape[::-1]
		found = None

		for scale in np.linspace(0.2, 1.0, 20)[::-1]:

			# # resize the image according to the scale, and keep track
			# # of the ratio of the resizing
			# resized = cv.resize(img_gray, width = int(img_gray.shape[1] * scale))
			# r = img_gray.shape[1] / float(resized.shape[1])


			# resize image
			scale_percent = scale
			width = int(img_gray.shape[1] * scale)
			height = int(img_gray.shape[0] * scale)
			dim = (width, height)
			resized = cv.resize(img_gray, dim, interpolation = cv.INTER_AREA)

			# if the resized image is smaller than the template, then break
			# from the loop
			# detect edges in the resized, grayscale image and apply template
			# matching to find the template in the image
			edged = cv.Canny(resized, 50, 200)
			result = cv.matchTemplate(edged, template, cv.TM_CCOEFF)
			(_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
			# if we have found a new maximum correlation value, then update
			# the found variable if found is None or maxVal > found[0]:
			if resized.shape[0] < h or resized.shape[1] < w:
				break
			found = (maxVal, maxLoc, r)

		# unpack the found variable and compute the (x, y) coordinates
		# of the bounding box based on the resized ratio
		(_, maxLoc, r) = found
		(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
		(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

		# draw a bounding box around the detected result and display the image
		cv.rectangle(img_rgb, (startX, startY), (endX, endY), (0, 0, 255), 2)
		cv.imshow("Image", img_rgb)
		cv.waitKey(0)

	def TestMethod(self):
		method = cv.TM_SQDIFF_NORMED
		large_image = cv.imread(self.image_path)
		small_image = cv.imread(os.path.join(self.GENERAL_PATH, 'sign', 'ucgen_real.png'))

		file1 = open(self.GetSubResultPath("textout.py"), "w", encoding="utf-8")
		result = cv.matchTemplate(small_image, large_image, method)
		print(result, file1, width = 10)
		# cv.imshow('output', result), cv.waitKey(0)
		plt.subplot(121),plt.imshow(result, cmap = 'gray')
		plt.show()

		sys.exit()