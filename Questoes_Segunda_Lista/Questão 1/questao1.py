import cv2
import numpy as np
from matplotlib import pyplot as plt

def harris(filename):
    imagem = cv2.imread(filename)
    gray = cv2.cvtColor(imagem,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst,None)

    imagem[dst>0.01*dst.max()]=[0,0,255]

    cv2.imwrite('harris-'+filename+".png" ,imagem)    


def shi_tomasi(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)

    cv2.imwrite('tomasi-'+filename+".png" ,img)   


def draw_corners(image, corners_map):
    for corner in corners_map:
        cv2.circle(image, (corner[1], corner[0]), 2, (0, 255, 0), -1)


def moravec(image, threshold=50):
    corners = []
    xy_shifts = [(1, 0), (1, 1), (0, 1), (-1, 1)]
    width, height = image.shape[:2]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            E = 10000000
            for shift in xy_shifts:
                diff = image[x + shift[0], y + shift[1]]
                diff = ((diff - image[x, y]) ** 2)
                if diff < E:
                    E = diff
            if E > threshold:
                corners.append((x, y))

    return corners



filenames=['questao1-img1.png', 'questao1-img2.jpg', 'questao1-img3.jpg', 'questao1-img4.jpg', 'questao1-img5.jpg', 'questao1-img6.jpg']

for filename in filenames:
    harris(filename)
    shi_tomasi(filename)

for image_path in filenames:
    threshold = 100

    image_RGB = cv2.imread(image_path)
    image = cv2.imread(image_path, 0)

    corners = moravec(image, threshold)
    draw_corners(image_RGB, corners)
    cv2.imwrite('moravec-'+image_path+".png" ,image_RGB)   