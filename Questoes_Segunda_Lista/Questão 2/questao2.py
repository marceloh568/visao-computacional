import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def sift(filename):
    img = cv.imread(filename)
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv.drawKeypoints(gray,kp,img)
    cv.imwrite('sift-'+filename  ,img)    

def surf(filename):
    img = cv.imread(filename, 0)
    surf = cv.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(img,None)
    img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
    plt.imshow(img2),plt.show()

def orb(filename):
    img = cv.imread(filename,0)
    orb = cv.ORB_create()
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()

filenames=['questao1-img1.png', 'questao1-img2.jpg', 'questao1-img3.jpg', 'questao1-img4.jpg', 'questao1-img5.jpg', 'questao1-img6.jpg']
for filename in filenames:
    sift(filename)
    surf(filename)
    orb(filename)
cv.waitKey(0)
cv.destroyAllWindows()