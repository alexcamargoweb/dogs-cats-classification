# Dogs/Cats classificator - https://github.com/alexcamargoweb/dogs-cats-classification
# Classificação de cachorros e gatos: uma RNA simplificada com Python e Keras.
# Adrian Rosebrock, A simple neural network with Python and Keras. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/.
# Acessado em: 16/02/2021.
# Arquivo: predict.py
# Execução via PyCharm/Linux (Python 3.8)
# $ conda activate tensorflow_keras

# importa os pacotes necessários
from __future__ import print_function
from keras.models import load_model
import numpy as np
import cv2

# função que converte uma imagem para um vetor
def image_to_feature_vector(image, size = (32, 32)):
    # redimensiona a imagem para um tamanho fixo e
    # realiza o flatten (achatamento/vetorização)
    return cv2.resize(image, size).flatten()

# caminho da imagem a ser classificada
IMAGE = './input/cat.jpg'
# diretório de entrada do modelo gerado
MODEL = './model'
# tamanho dos lotes passados a rede
BATCH = 32

# inicializa os rótulos com base no dataset
CLASSES = ["cat", "dog"]
# carrega a rede
print("[INFO] carregando a arquitetura da rede e seus pesos...")
model = load_model(MODEL)

print("[INFO] classificando...")
# carrega a imagem
image = cv2.imread(IMAGE)
# redimensiona a imagem para a exibição
image = cv2.resize(image, (400, 400))
# extrai as caracteristicas da imagem
features = image_to_feature_vector(image) / 255.0
features = np.array([features])

# classifica a imagem usando a rede pré-treinada
probs = model.predict(features)[0]
prediction = probs.argmax(axis = 0)

# imprime a classe e a probabilidade na imagem de saída
label = "{}: {:.2f}%".format(CLASSES[prediction], probs[prediction] * 100)
cv2.putText(image, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
cv2.imshow("Dogs/Cats classificator", image)
cv2.waitKey(0)
