import cv2
import numpy as np
from scipy import ndimage
 
kernel_3x3 = np.array([[-1,-1,-1],
                       [-1, 8,-1],
                       [-1,-1,-1]])
kernel_5x5 = np.array([[-1,-1,-1,-1,-1],
                       [-1, 1, 2, 1,-1],
                       [-1, 2, 4, 2,-1],
                       [-1, 1, 2, 1,-1],
                       [-1,-1,-1,-1,-1]])
 
imagem = cv2.imread('mulher_pb.png', 0)
imagem_3x3 = ndimage.convolve(imagem, kernel_3x3) 
imagem_5x5 = ndimage.convolve(imagem, kernel_5x5)

cv2.imwrite( 'questao5-3x3.png', imagem_3x3 )
cv2.imwrite( 'questao5-5x5.png', imagem_5x5 )