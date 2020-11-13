import cv2
import numpy

def filtragem_bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):

    gau = lambda r2, sigma: (numpy.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    win_x = int( 3*sigma_s+1 )

    w_soma = numpy.ones( img_in.shape )*reg_constant
    resultado  = img_in*reg_constant

    for shft_x in range(-win_x,win_x+1):
        for shft_y in range(-win_x,win_x+1):
            w = gau( shft_x**2+shft_y**2, sigma_s )

            off = numpy.roll(img_in, [shft_y, shft_x], axis=[0,1] )
            tw = w*gau( (off-img_in)**2, sigma_v )

            resultado += off*tw
            w_soma += tw
    return resultado/w_soma

imagem = cv2.imread('first_frame.jpg', cv2.IMREAD_UNCHANGED ).astype(numpy.float32)/255.0

zb = numpy.stack([ 
        filtragem_bilateral( imagem[:,:,0], 10.0, 0.1 ),
        filtragem_bilateral( imagem[:,:,1], 10.0, 0.1 ),
        filtragem_bilateral( imagem[:,:,2], 10.0, 0.1 )], axis=2 )

O = numpy.hstack( [imagem,zb] )
cv2.imwrite( 'questao3.png', O*255.0 )