import cv2
import numpy as np


imagem = cv2.imread('mulher.png')
mediano = cv2.medianBlur(imagem, 5)
gau = cv2.GaussianBlur(imagem, (5,5), 0)

comparacao = np.concatenate((mediano, gau), axis=1)


cv2.imwrite( 'questao4-medianBlur.png', mediano )
cv2.imwrite( 'questao4-gausBlur.png', gau )
cv2.imwrite( 'questao4-camparacao.png', comparacao )