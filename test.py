# Dogs/Cats classificator - https://github.com/alexcamargoweb/dogs-cats-classification
# Classificação de cachorros e gatos: uma RNA simplificada com Python e Keras.
# Adrian Rosebrock, A simple neural network with Python and Keras. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/.
# Acessado em: 16/02/2021.
# Arquivo: test.py
# Execução via PyCharm/Linux (Python 3.8)
# $ conda activate tensorflow_keras

# importa os pacotes necessários
from __future__ import print_function
from keras.models import load_model
from imutils import paths
import numpy as np
import cv2

# função que converte uma imagem para um vetor
def image_to_feature_vector(image, size=(32, 32)):
    # redimensiona a imagem para um tamanho fixo e
    # realiza o flatten (achatamento/vetorização)
    return cv2.resize(image, size).flatten()

# caminho das imagens de teste
DATASET = './dataset/test'
# diretório de entrada do modelo gerado
MODEL = './model'
# tamanho dos lotes passados a rede
BATCH = 32

# inicializa os rótulos com base no dataset
CLASSES = ["cat", "dog"]
# carrega a rede
print("[INFO] carregando a arquitetura da rede e seus pesos...")
model = load_model(MODEL)
print("[INFO] testando as imagens de {}".format(DATASET))

# faz um loop sobre as imagens de teste
for imagePath in paths.list_images(DATASET):
    # carrega a imagem e redimensiona para um tamanho fixo de 32 x 32 píxels
    print("[INFO] classificando {}".format(imagePath[imagePath.rfind("/") + 1:]))
    image = cv2.imread(imagePath)
    # extrai as caracteristicas da imagem
    features = image_to_feature_vector(image) / 255.0
    features = np.array([features])

    # classifica a imagem usando a rede pré-treinada
    probs = model.predict(features)[0]
    prediction = probs.argmax(axis = 0)
    # imprime a classe e a probabilidade na imagem de saída
    label = "{}: {:.2f}%".format(CLASSES[prediction], probs[prediction] * 100)
    cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    # exibe a imagem
    cv2.imshow("Dogs/Cats classificator", image)
    cv2.waitKey(0)