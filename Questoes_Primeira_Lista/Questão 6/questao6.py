import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

imagem = cv.imread('mulher_pb.png')
kernel_5x5 = np.ones((5,5),np.float32)/25
imagem_5x5 = cv.filter2D(imagem,-1,kernel_5x5)

# 3x3
kernel_3x3 = np.ones((3,3),np.float32)/25
imagem_3x3 = cv.filter2D(imagem,-1,kernel_3x3)

cv.imwrite("questao6-5x5.png", imagem_5x5)
cv.imwrite("questao6-3x3.png", imagem_3x3)