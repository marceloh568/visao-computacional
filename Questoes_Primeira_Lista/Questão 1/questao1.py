from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

imagemA = cv2.imread("first_frame.jpg")
imagemB = cv2.imread("second_frame.jpg")

cinzaA = cv2.cvtColor(imagemA, cv2.COLOR_BGR2GRAY)
cinzaB = cv2.cvtColor(imagemB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(cinzaA, cinzaB, full=True)
diff = (diff * 255).astype("uint8")

thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contornos = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
contornos = imutils.grab_contours(contornos)

for c in contornos:
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imagemA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imagemB, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow("Thresh", thresh)
cv2.imwrite('questao1.png', thresh)
cv2.waitKey(0)