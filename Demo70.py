import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
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

model1 = Sequential()
model1.add(Dense(units=128, activation=tf.nn.relu, input_shape=(flattenDim,)))
model1.add(Dense(units=64, activation=tf.nn.relu))
model1.add(Dense(units=10, activation=tf.nn.softmax))
model1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model1.summary())
accuracyCallback = TensorBoard(log_dir="logs", histogram_freq=0, write_graph=True,
                               write_images=True)
model1.fit(trainImages, trainLabels, epochs=20, callbacks=[accuracyCallback])

predictLabels = model1.predict_classes(testImages)
print("result=", predictLabels[:20])

loss, accuracy = model1.evaluate(testImages, testLabels)
print('Test accuracy: %.4f' % (accuracy))

# (pykt0803)
# cd C:\Users\Admin\PycharmProjects\PYKT0803
# tensorboard --logdir=logs