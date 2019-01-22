from os import listdir
from PIL import Image
import numpy as numpy

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 243


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


data_set = open_files()
average_matrix = get_average(data_set)


