import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc
from scipy import ndimage
import math

def escala_espectro(A):
   return numpy.real(numpy.log10(numpy.absolute(A) + numpy.ones(A.shape)))

def aplicaFiltroGaussiano(numRows, numCols, sigma, highPass=True):
   centroI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
   centroJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)

   def gaussiano(i,j):
      coeficiente = math.exp(-1.0 * ((i - centroI)**2 + (j - centroJ)**2) / (2 * sigma**2))
      return 1 - coeficiente if highPass else coeficiente

   return numpy.array([[gaussiano(i,j) for j in range(numCols)] for i in range(numRows)])


def filtroDFT(imagemDaMatrix, filterMatrix):
   shiftedDFT = fftshift(fft2(imagemDaMatrix))

   filtragemDFT = shiftedDFT * filterMatrix
   return ifft2(ifftshift(filtragemDFT))


def lowPass(imagemDaMatrix, sigma):
   n,m = imagemDaMatrix.shape
   return filtroDFT(imagemDaMatrix, aplicaFiltroGaussiano(n, m, sigma, highPass=False))


def highPass(imagemDaMatrix, sigma):
   n,m = imagemDaMatrix.shape
   return filtroDFT(imagemDaMatrix, aplicaFiltroGaussiano(n, m, sigma, highPass=True))


def imagemHibrida(highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
   highPassed = highPass(highFreqImg, sigmaHigh)
   lowPassed = lowPass(lowFreqImg, sigmaLow)

   return highPassed + lowPassed


einstein = ndimage.imread("einstein.png", flatten=True)
marilyn = ndimage.imread("marilyn.png", flatten=True)

van_go = ndimage.imread("van_go.png", flatten=True)
frida_k = ndimage.imread("frida_k.png", flatten=True)

jim_mo = ndimage.imread("jim_mo.png", flatten=True)
che_gue = ndimage.imread("che_gue.png", flatten=True)

for i in range(0,30, 10):
   hybrid = imagemHibrida(einstein, marilyn, 25 + i, 10 + i)
   misc.imsave("questao8-imagem_einstein_marilyn" + str(25 + i) + "x" + str(10+i) +".png", numpy.real(hybrid))

   hybridVanFri = imagemHibrida(van_go, frida_k, 25 + i, 10 + i)
   misc.imsave("questao8-imagem_van_frida" + str(25 + i) + "x" + str(10+i) +".png", numpy.real(hybridVanFri))

   hybridJimChe = imagemHibrida(jim_mo, che_gue, 25 + i, 10 + i)
   misc.imsave("questao8-jim_che" + str(25 + i) + "x" + str(10+i) +".png", numpy.real(hybridJimChe))