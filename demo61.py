import numpy as np
import tensorflow as tf
scores = [3.0, 1.0, 2.0]


def manualSoftmax(x):
    arrayX = np.array(x)
    return np.exp(arrayX) / np.sum(np.exp(arrayX), axis=0)


def normalRatio(x):
    arrayX = np.array(x)
    return arrayX / np.sum(arrayX, axis=0)

print(normalRatio(scores))
print(manualSoftmax(scores))

print(tf.nn.softmax(scores).numpy())