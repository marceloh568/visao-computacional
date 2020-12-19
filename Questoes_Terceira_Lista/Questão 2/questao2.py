from imutils import paths
import numpy as np
import imutils
import cv2


imagePaths = [
    ['1_esquerda.jpg', '1_meio.jpg', '1_direita.jpg'],
    ['2_esquerda.jpg', '2_meio.jpg', '2_direita.jpg'],
    ['3_esquerda.jpg', '3_meio.jpg', '3_direita.jpg'],
    ['4_esquerda.jpg', '4_meio.jpg', '4_direita.jpg'],
    ['5_esquerda.jpg', '5_meio.jpg', '5_direita.jpg'],
]

images = []

for imagePath in imagePaths:
	for imageName in imagePath:
		image = cv2.imread(imageName)
		images.append(image)

	stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
	(status, stitched) = stitcher.stitch(images)

	if status == 0:
		stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
		gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)
		mask = np.zeros(thresh.shape, dtype="uint8")
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
		
		minRect = mask.copy()
		sub = mask.copy()
		while cv2.countNonZero(sub) > 0:
			minRect = cv2.erode(minRect, None)
			sub = cv2.subtract(minRect, thresh)

		cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)
		(x, y, w, h) = cv2.boundingRect(c)
		stitched = stitched[y:y + h, x:x + w]
		cv2.imshow("result.jpg", stitched)
		images = []
		cv2.waitKey(0)