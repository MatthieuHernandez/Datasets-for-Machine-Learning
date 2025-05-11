# This script display images from MNIST.
import struct
import numpy as np
import matplotlib.pyplot as plt
import sys


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f'Invalid magic number {magic} in MNIST image file: {filename}')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)
        images = images.reshape((num_images, rows, cols))
        return images


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f'Invalid magic number {magic} in MNIST label file: {filename}')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels


if __name__ == '__main__':
    try:
        index = int(sys.argv[2])
    except ValueError:
        print('Invalid arg.')
        sys.exit(1)

    images = load_mnist_images(f'.\\{sys.argv[1]}-images.idx3-ubyte')
    labels = load_mnist_labels(f'.\\{sys.argv[1]}-labels.idx1-ubyte')
    first_image = images[index]
    label = labels[index]
    plt.imshow(first_image, cmap='gray')
    plt.axis('off')
    plt.title(f"MNIST - Label: {label}")
    plt.show()