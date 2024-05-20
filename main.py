import cv2
import numpy as np

# implementação de um sistema que leia uma imagem contendo um número qualquer de dados
# e contabilize a soma das faces destes. 

# função de processamento da imagem com filtros, detecção de contornos e e exibição da imagem e pontuação dos dados
def processar_imagem(imagem_path):

    # carrega a imagem conforme caminho informado
    imagem = cv2.imread(imagem_path)

    # converte a imagem para tons de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # aplica um filtro de limiarização para segmentar os objetos brancos
    _, limiarizada = cv2.threshold(imagem_cinza, 217, 255, cv2.THRESH_BINARY)

    # aplica uma operação de abertura para remover os pontos pequenos
    kernel = np.ones((7, 7), np.uint8)
    limiarizada = cv2.morphologyEx(limiarizada, cv2.MORPH_OPEN, kernel)

    # detecta contornos na imagem limiarizada
    contornos, _ = cv2.findContours(limiarizada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # inicializa um contador de pontos
    pontos = 0

    # cria uma máscara do mesmo tamanho da imagem original
    mascara = np.zeros(imagem.shape[:2], dtype="uint8")

    # rotina para detecção e contorno de cada ponto
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        perimetro = cv2.arcLength(contorno, True)

        # checa se o perímetro é maior que zero para evitar divisão por zero
        if perimetro > 0:
            circularidade = 4 * np.pi * area / (perimetro * perimetro)

            # refina os valores de tamanho de circularidade para definir mais os pontos detectados
            if 0.85 < circularidade < 1.5:  
                # desenho os contornos verdes na máscara 
                cv2.drawContours(imagem, [contorno], -1, (0, 255, 0), thickness=2)
                pontos += 1

    # aplica a máscara à imagem original
    imagem_limpa = cv2.bitwise_and(imagem, imagem, mask=mascara)

    # exibe a pontuação total de cada imagem
    print("Pontuação total da imagem atual:", pontos)

    # exibe a imagem com os círculos contornados
    cv2.imshow('Imagem com pontos contabilizados', imagem)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    return pontos

# inicializa a pontuação geral
pontuacao_geral = 0

# processa quatro imagens diferentes e soma todos os pontos
imagens_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
for path in imagens_paths:
    pontos_imagem = processar_imagem(path)

    # incrementa a pontuação da imagem atual à pontuação geral
    pontuacao_geral += pontos_imagem  

# exibe a pontuação geral
print("Pontuação geral de todas as imagens:", pontuacao_geral)