import cv2
import numpy as np

def supressao_nao_maxima(caixas_limites, pontuacao, threshold):
    if len(caixas_limites) == 0:
        return [], []

    caixas = np.array(caixas_limites)

    inicio_x = caixas[:, 0]
    inicio_y = caixas[:, 1]
    fim_x = caixas[:, 2]
    fim_y = caixas[:, 3]

    score = np.array(pontuacao)

    caixas_escolhidas = []
    pontuacao_escolhida = []

    areas = (fim_x - inicio_x + 1) * (fim_y - inicio_y + 1)

    order = np.argsort(score)

    while order.size > 0:
        index = order[-1]

        caixas_escolhidas.append(caixas_limites[index])
        pontuacao_escolhida.append(pontuacao[index])

        x1 = np.maximum(inicio_x[index], inicio_x[order[:-1]])
        x2 = np.minimum(fim_x[index], fim_x[order[:-1]])
        y1 = np.maximum(inicio_y[index], inicio_y[order[:-1]])
        y2 = np.minimum(fim_y[index], fim_y[order[:-1]])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        esquerda = np.where(ratio < threshold)
        order = order[esquerda]

    return caixas_escolhidas, pontuacao_escolhida


caixas_limites = [(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304)]
pontuacao = [0.9, 0.75, 0.8]

imagem = cv2.imread('cat.png')

org = imagem.copy()

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2

threshold = 0.4

for (inicio_x, inicio_y, fim_x, fim_y), confidence in zip(caixas_limites, pontuacao):
    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
    cv2.rectangle(org, (inicio_x, inicio_y - (2 * baseline + 5)), (inicio_x + w, inicio_y), (0, 255, 255), -1)
    cv2.rectangle(org, (inicio_x, inicio_y), (fim_x, fim_y), (0, 255, 255), 2)
    cv2.putText(org, str(confidence), (inicio_x, inicio_y), font, font_scale, (0, 0, 0), thickness)

caixas_escolhidas, pontuacao_escolhida = supressao_nao_maxima(caixas_limites, pontuacao, threshold)

for (inicio_x, inicio_y, fim_x, fim_y), confidence in zip(caixas_escolhidas, pontuacao_escolhida):
    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
    cv2.rectangle(imagem, (inicio_x, inicio_y - (2 * baseline + 5)), (inicio_x + w, inicio_y), (0, 255, 255), -1)
    cv2.rectangle(imagem, (inicio_x, inicio_y), (fim_x, fim_y), (0, 255, 255), 2)
    cv2.putText(imagem, str(confidence), (inicio_x, inicio_y), font, font_scale, (0, 0, 0), thickness)

cv2.imshow('Original', org)
cv2.imshow('supressao_nao_maxima', imagem)
cv2.waitKey(0)