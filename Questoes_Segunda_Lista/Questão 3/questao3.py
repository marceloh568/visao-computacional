import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def orbComForcaBruta(primeiraImagem,segundaImagem):
    img1 = cv.imread(primeiraImagem,cv.IMREAD_GRAYSCALE)          
    img2 = cv.imread(segundaImagem,cv.IMREAD_GRAYSCALE) 
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

def siftComForcaBruta(primeiraImagem,segundaImagem):
    img1 = cv.imread(primeiraImagem,cv.IMREAD_GRAYSCALE)          
    img2 = cv.imread(segundaImagem,cv.IMREAD_GRAYSCALE)
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def combinadorFlann(primeiraImagem, segundaImagem):
    img1 = cv.imread(primeiraImagem,cv.IMREAD_GRAYSCALE)          
    img2 = cv.imread(segundaImagem,cv.IMREAD_GRAYSCALE) 
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()


filenames=['1.PNG', '2.PNG', '3.PNG', '4.PNG', '5.PNG', '6.PNG']

orbComForcaBruta(filenames[0], filenames[1])
orbComForcaBruta(filenames[1], filenames[2])
orbComForcaBruta(filenames[2], filenames[3])

siftComForcaBruta(filenames[0], filenames[1])
siftComForcaBruta(filenames[1], filenames[2])
siftComForcaBruta(filenames[2], filenames[3])

combinadorFlann(filenames[0], filenames[1])
combinadorFlann(filenames[1], filenames[2])
combinadorFlann(filenames[2], filenames[3])