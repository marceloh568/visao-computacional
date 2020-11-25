import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from scipy import ndimage as ndi
import matplotlib.pyplot as plt

def filtroLog(imagem, escalas):
    img = cv2.imread(imagem)
    for i in escalas:
        blur = cv2.GaussianBlur(img,(i,i),0)
        laplacian = cv2.Laplacian(blur,cv2.CV_64F)
        laplacian1 = laplacian/laplacian.max()
        maximoLocalPosicaoEscala(laplacian1)

def maximoLocalPosicaoEscala(img):
    im = img_as_float(img)
    image_max = ndi.maximum_filter(im, size=20, mode='constant')

    coordinates = peak_local_max(im, min_distance=20)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(image_max, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Máximo Local')

    ax[2].imshow(im, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[2].axis('off')
    ax[2].set_title('PLM')

    fig.tight_layout()

    plt.show()

filtroLog('bac.jpg', [1, 3, 5, 7, 9, 11])