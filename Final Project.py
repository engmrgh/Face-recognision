from os import listdir
from PIL import Image
import numpy as numpy
from sys import getsizeof

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 243
SIZE = 243
NUMBER_OF_EIGENVECTORS = 6
NUMBER_OF_IMAGES = 165

#Section 1
def open_files():
    file_path = "./yalefaces/"
    images = []
    for file in listdir(file_path):
        images.append(numpy.asarray(Image.open(file_path + file)))
    return images


def get_average(data_set):
    psi = numpy.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for image in data_set:
        psi = numpy.add(psi, image)
    psi = (1/len(data_set))*psi
    return psi


def save_average_matrix(psi):
    img = Image.fromarray(psi)
    if img.mode != 'L':
        img = img.convert('L')
    img.save("./Constructed Photos/Section 1/Psi/Psi.png")


def get_subtract_matrix(dataset, psi):
    Q = []
    for image in dataset:
        Q.append(image - psi)
    return Q


def save_subtract_matrix(Q):
    for img, i in zip(Q, range(len(Q))):
        img = Image.fromarray(img)
        if img.mode != 'L':
            img = img.convert('L')
            img.save("./Constructed Photos/Section 1/Q/" + str(i) + ".png")


def get_transposed_subtract_matrix(subtract_matrix):
    QT = []
    for image in subtract_matrix:
        QT.append(image.transpose())
    return QT


def save_transposed_subtract_matrix(QT):
    for img, i in zip(QT, range(len(QT))):
        img = Image.fromarray(img)
        if img.mode != 'L':
            img = img.convert('L')
            img.save("./Constructed Photos/Section 1/QT/" + str(i) + ".png")


def get_covariance_matrix(q, qt):
    C = [[0 for i in range(SIZE)] for j in range(SIZE)]
    for image, imageT in zip(q, qt):
        C = numpy.add(numpy.matmul(image, imageT), C)
    C = (1/len(q)) * C
    return C


def save_covariance_matrix(C):
    img = Image.fromarray(C)
    if img.mode != 'L':
        img = img.convert('L')
    img.save("./Constructed Photos/Section 1/C/C.png")


def get_eigenvalues_and_vectors(C):
    eigenvalues, eigenvctrs = numpy.linalg.eig(C)
    sorted_eigenvalues = sorted(eigenvalues)
    return_vectors = []
    return_values = []
    for i in range(6):
        eigenvalue = sorted_eigenvalues[len(eigenvalues) - i - 1]
        for j in range(len(eigenvalues)):
            if eigenvalues[j] == eigenvalue:
                return_vectors.append(eigenvctrs[j])
                return_values.append(eigenvalue)
    return return_values, return_vectors


def calculate_f(eigenvctrs,Q):
    F = []
    for i in range(NUMBER_OF_EIGENVECTORS):
        result = 0
        for k in range(NUMBER_OF_IMAGES):
            result += eigenvctrs[i][k] * Q[k]
        F.append(result)
    return F

#code place
data_set = open_files()
average_matrix = get_average(data_set)
save_average_matrix(average_matrix)
subtract_matrix = get_subtract_matrix(data_set, average_matrix)
save_subtract_matrix(subtract_matrix)
transposed_subtract_matrix = get_transposed_subtract_matrix(subtract_matrix)
save_transposed_subtract_matrix(transposed_subtract_matrix)
covariance_matrix = get_covariance_matrix(subtract_matrix, transposed_subtract_matrix)
save_covariance_matrix(covariance_matrix)
eigenvalues, eigenvectors = get_eigenvalues_and_vectors(covariance_matrix)
matrix_F = calculate_f(eigenvectors, subtract_matrix)


#Section 2 SVD Compression
data_path = "./yalefaces/subject05.centerlight.gif"
save_path = "./Constructed Photos/Section 2/SVD Compression/"
image = numpy.asarray(Image.open(data_path))
U, sigma, VT = numpy.linalg.svd(image)
for i in range(5, 130, 25):
    tmp_sigma = numpy.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
    for j in range(i):
        tmp_sigma[j][j] = sigma[j]
    compressed = numpy.matmul(numpy.matmul(U, tmp_sigma), VT)
    img = Image.fromarray(compressed)
    if img.mode != 'L':
        img = img.convert('L')
    img.save(save_path + str(i) + ".png")


# Section 3 Face recognition with SVD


def get_matrix_s():
    training_data_path = "./training data/"
    S = []
    for file in listdir(training_data_path):
        a = numpy.asarray(Image.open(training_data_path + file)).flatten()
        S.append(a)
    return S


def get_matrix_F_bar(S):
    f_bar = numpy.zeros((IMAGE_WIDTH * IMAGE_HEIGHT, 1))
    S_2 = []
    for f in S:
        tmp = (1/len(S)) * f
        S_2.append(tmp)
    f_bar = S_2[0] + S_2[1] + S_2[2] + S_2[3] + S_2[4] + S_2[5]
    return f_bar


matrix_S = get_matrix_s()
matrix_F_bar = get_matrix_F_bar(matrix_S)

