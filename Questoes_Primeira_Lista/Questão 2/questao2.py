import cv2 
import numpy as np 

imagem = cv2.imread('questao1.png') 
cv2.waitKey(0) 

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) 

bordas = cv2.Canny(cinza, 5, 200) 
cv2.waitKey(0) 

contornos, h = cv2.findContours(bordas, 
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

cv2.drawContours(imagem, contornos, -1, (0, 0, 255), 3) 

cv2.imwrite('questao2.png', imagem)
cv2.imshow('Resultado', imagem) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 