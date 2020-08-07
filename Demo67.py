import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(f"train_iamges shape={train_images.shape}, test_images shape={test_images.shape}")
print(f"train label len={len(train_labels)}, test label len={len(test_labels)}")
print(np.unique(train_labels))


def plotImage(index):
    plt.title(f"training image [{index}] marked as {train_labels[index]}")
    plt.imshow(train_images[index], cmap='binary')
    plt.show()

plotImage(567)


def plotTestImage(index):
    plt.title(f"testimg image [{index}] marked as {test_labels[index]}")
    plt.imshow(test_images[index], cmap='binary')
    plt.show()
# plotTestImage(182)