from os import listdir
from PIL import Image
import numpy as numpy

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 243
SIZE = 243
NUMBER_OF_EIGENVECTORS = 6
NUMBER_OF_IMAGES = 165


def open_files():
    file_path = "./yalefaces/"
    images = []
    for file in listdir(file_path):
        images.append(numpy.asarray(Image.open(file_path + file)))
    return images


def get_average(data_set):
    psi = [[0 for i in range(IMAGE_WIDTH)] for j in range(IMAGE_HEIGHT)]
    for image in data_set:
        psi = numpy.add(psi, image)
    psi = (1/len(data_set))*psi
    return psi


def save_average_matrix(psi):
    img = Image.fromarray(psi)
    if img.mode != 'L':
        img = img.convert('L')
    img.save("./Constructed Photos/Section 1/Psi/Psi.png")


def get_subtract_matrix(dataset,psi):
    Q = []
    for image in data_set:
        Q.append(numpy.subtract(image,psi))
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
matrix_F = calculate_f(eigenvectors,subtract_matrix)
print(matrix_F)


