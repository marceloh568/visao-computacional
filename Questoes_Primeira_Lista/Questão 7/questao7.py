import sys
import cv2 as cv


def arestas(title, imagemName):
    ddepth = cv.CV_16S
    kernel_size = 3
    src = cv.imread(cv.samples.findFile(imagemName), cv.IMREAD_COLOR)

    src = cv.GaussianBlur(src, (3, 3), 0)
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
    abs_dst = cv.convertScaleAbs(dst)
    cv.imwrite("questao7-"+ title +".png", abs_dst)


arestas("viralata", "vira-caramelo.jpg")
arestas("dado", "dado.jpg")
arestas("cadeira", "cadeira.jpeg")