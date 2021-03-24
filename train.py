# Dogs/Cats classificator - https://github.com/alexcamargoweb/dogs-cats-classification
# Classificação de cachorros e gatos: uma RNA simplificada com Python e Keras.
# Adrian Rosebrock, A simple neural network with Python and Keras. PyImageSearch.
# Disponível em: https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/.
# Acessado em: 16/02/2021.
# Arquivo: train.py
# Execução via PyCharm/Linux (Python 3.8)
# $ conda activate tensorflow_keras

# importa os pacotes necessários
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import cv2
import os

# função que converte uma imagem para um vetor
def image_to_feature_vector(image, size = (32, 32)):
    # redimensiona a imagem para um tamanho fixo e
    # realiza o flatten (achatamento/vetorização)
    return cv2.resize(image, size).flatten()

# caminho das imagens de treinamento
DATASET = './dataset/train'
# diretório de saída do modelo gerado
MODEL = './model'

# armazena numa lista as imagens do dataset
print("[INFO] carregando images...")
imagePaths = list(paths.list_images(DATASET))
data = []  # dados
labels = []  # rótulos

# imprime a quantidade de imagens do dataset
print("TOTAL:" + str(len(imagePaths)))

# faz um loop sobre as imagens de entrada
for (i, imagePath) in enumerate(imagePaths):
    # carrega a imagem e extrai o rótulo
    # formato de entrada: {class}.{image_num}.jpg
    image = cv2.imread(imagePath)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]
    # constrói um vetor de pixels brutos
    features = image_to_feature_vector(image)
    # atualiza a matriz de dados e a lista de rótulos
    data.append(features)
    labels.append(label)
    # exibe o carregamento e atualiza a cada 100 imagens
    if i > 0 and i % 100 == 0:
        print("[INFO] processando imagens: {}/{}".format(i, len(imagePaths)))

# codifica os rótulos, convertendo-os em inteiros
le = LabelEncoder()
labels = le.fit_transform(labels)
# dimensiona os pixels da imagem de entrada para o intervalo [0, 1]
data = np.array(data) / 255.0
# transforma os rótulos em vetores no intervalo [0, num_classes]
labels = np_utils.to_categorical(labels, 2)
# particiona os dados em divisões de treinamento e teste, usando 75%
# dos dados para treinamento e os 25% restantes para teste
print("\n[INFO] dividindo o dataset em treino/teste...")
(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels,
                                                                  test_size = 0.25,
                                                                  random_state = 777)
# define a arquitetura da rede
model = Sequential()
# camada de entrada com 3072 inputs: 32 x 32 x 3 = 3,072 píxels brutos
model.add(Dense(768, input_dim = 3072, kernel_initializer = 'uniform', activation = "relu"))
model.add(Dense(384, kernel_initializer = "uniform", activation = "relu"))
# camada de saída: "dog" ou "cat"
model.add(Dense(2))
# função de ativação: retorna a probabilidade para cada classe
model.add(Activation("softmax"))

# TRAIN

# treina o modelo usando o SGD (Stochastic Gradient Descent)
print("[INFO] compilando o modelo...\n")
sgd = SGD(lr = 0.01)
model.compile(loss = "binary_crossentropy", optimizer = sgd, metrics = ["accuracy"])
model.fit(trainData, trainLabels, epochs = 50, batch_size = 128, verbose = 1)

# TEST

# exibe a acurácia do conjunto de teste
print("\n[INFO] avaliando o conjunto de teste...")
(loss, accuracy) = model.evaluate(testData, testLabels, batch_size = 128, verbose = 1)
print("[INFO] loss = {:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))
# salva a arquitetura e os pesos no arquivo
print("\n[INFO] salvando a arquitetura e os pesos da rede...\n")
model.save(MODEL)
