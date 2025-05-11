# This script convert matlab files from https://www.cs.toronto.edu/~tijmen/affNIST/ to MNIST data format.
from scipy.io import loadmat
import struct
import numpy as np
import sys


def create_new_image_file(name, images, num_images):
    header = struct.pack('>IIII', 2051, num_images, 40, 40)
    with open(name, 'wb') as new_file:
        new_file.write(header)
        new_file.write(images.tobytes())


def create_new_index_file(name, index):
    header = struct.pack('>II', 2049, index.shape[0])
    with open(name, 'wb') as new_file:
        new_file.write(header)
        new_file.write(index.tobytes())


def convert_file(index):
    file = loadmat(f'.\\training_and_validation_batches\\{index}.mat')
    images = file['affNISTdata'][0][0][2].transpose()
    image_index =  file['affNISTdata'][0][0][5].astype(np.int8)
    create_new_image_file('train-images.idx3-ubyte', images.reshape(-1), 60000)
    create_new_index_file('train-labels.idx1-ubyte', image_index)

    file = loadmat(f'.\\test_batches\\{index}.mat')
    images = file['affNISTdata'][0][0][2].transpose()
    image_index =  file['affNISTdata'][0][0][5].astype(np.int8)
    create_new_image_file('t10k-images.idx3-ubyte', images.reshape(-1), 10000)
    create_new_index_file('t10k-labels.idx1-ubyte', image_index)


if __name__ == '__main__':
    convert_file(1)