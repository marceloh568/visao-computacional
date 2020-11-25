import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def orbComForcaBruta(primeiraImagem,segundaImagem, qtdCorrespondencia):
    img1 = cv.imread(primeiraImagem,cv.IMREAD_GRAYSCALE)          
    img2 = cv.imread(segundaImagem,cv.IMREAD_GRAYSCALE) 
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:qtdCorrespondencia],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()
    cv.imwrite(primeiraImagem+str(qtdCorrespondencia)+'-correspondencias-'+segundaImagem+'.png',img3)


filenames=['1.PNG', '2.PNG']
orbComForcaBruta(filenames[0], filenames[1], 10)
orbComForcaBruta(filenames[0], filenames[1], 3)
orbComForcaBruta(filenames[0], filenames[1], 20)