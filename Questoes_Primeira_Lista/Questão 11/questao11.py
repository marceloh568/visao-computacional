import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def concatenar_duas_imagens(primeira_imagem, segunda_imagem):
    ha,wa = primeira_imagem.shape[:2]
    hb,wb = segunda_imagem.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    nova_imagem = np.zeros(shape=(max_height, total_width, 3))
    nova_imagem[:ha,:wa]=primeira_imagem
    nova_imagem[:hb,wa:wa+wb]=segunda_imagem
    return nova_imagem

def concatenar_varias_imagens(lista_imagens):
    resultado = None
    for i, caminho_imagem in enumerate(lista_imagens):
        imagem = plt.imread(caminho_imagem)[:,:,:3]
        if i==0:
            resultado = imagem
        else:
            resultado = concatenar_duas_imagens(resultado, imagem)
    return resultado

imagens = []

src = cv.imread(cv.samples.findFile('borb.png'))

for i in range(0,3):     
    rows, cols, _channels = map(int, src.shape)
    src = cv.pyrDown(src, dstsize=(cols // 2, rows // 2))
    cv.imwrite("pds"+str(i)+".png", src)
    imagens.append("pds"+str(i)+".png")


for i in range(0,3):
    rows, cols, _channels = map(int, src.shape)
    src = cv.pyrUp(src, dstsize=(2 * cols, 2 * rows))
    cv.imwrite("pg"+str(i)+".png", src)
    imagens.append("pg"+str(i)+".png")



imagens = [imagens[0], imagens[1], imagens[2], imagens[3], imagens[4], imagens[5]]
resultado = concatenar_varias_imagens(imagens)

plt.imshow(resultado)

plt.show()

cv.waitKey()
cv.destroyAllWindows()