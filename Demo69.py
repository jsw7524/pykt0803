import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

flattenDim = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)

trainImages = np.reshape(train_images, (TRAINING_SIZE, flattenDim))
testImages = np.reshape(test_images, (TEST_SIZE, flattenDim))
print(type(trainImages[0][0]))
trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
print(type(trainImages[0][0]))
trainImages /= 255
testImages /= 255
NUM_DIGITS = 10
trainLabels = to_categorical(train_labels, NUM_DIGITS)
testLabels = to_categorical(test_labels, NUM_DIGITS)
print(trainImages[0])
print(trainLabels[0])