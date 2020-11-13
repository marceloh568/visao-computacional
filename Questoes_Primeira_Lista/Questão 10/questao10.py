import cv2

def redimensionar_imagem(x, y, title, nomeImagem="fish.jpg"):
    imagem = cv2.imread(nomeImagem,cv2.IMREAD_COLOR)
    height, width, depth = imagem.shape
    ImagemNova = cv2.resize(imagem,(int(x),int(y)))
    cv2.imshow("Resultado",ImagemNova)
    cv2.waitKey(0)
    cv2.imwrite("questao10-"+title+".jpg",ImagemNova)

redimensionar_imagem(100,100, "peixe")
redimensionar_imagem(100,100, "cadeira", "cadeira.jpeg")
redimensionar_imagem(100,100, "dado", "dado.jpg")