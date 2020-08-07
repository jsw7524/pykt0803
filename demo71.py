import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard, ModelCheckpoint

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


def createModel():
    model = Sequential()
    model.add(Dense(units=128, activation=tf.nn.relu, input_shape=(flattenDim,)))
    model.add(Dense(units=64, activation=tf.nn.relu))
    model.add(Dense(units=10, activation=tf.nn.softmax))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


model1 = createModel()
# callback1 , TensorBoard
accuracyCallback = TensorBoard(log_dir="logs", histogram_freq=0, write_graph=True,
                               write_images=True)
checkPointCallback = ModelCheckpoint("ckpt/demo68.ckpt", save_weights_only=True, verbose=1)
model1.fit(trainImages, trainLabels, epochs=20, callbacks=[accuracyCallback, checkPointCallback])

predictLabels = model1.predict_classes(testImages)
print("[model1]result=", predictLabels[:20])

loss, accuracy = model1.evaluate(testImages, testLabels)
print('[model1]Test accuracy: %.4f' % (accuracy))

model2 = createModel()
loss2, accuracy2 = model2.evaluate(testImages, testLabels, verbose=2)
print('[mode2]test accuracy:%.4f' % (accuracy2))
model2.load_weights("ckpt/demo68.ckpt")
loss3, accuracy3 = model2.evaluate(testImages, testLabels, verbose=2)
print('[mode3]test accuracy:%.4f' % (accuracy3))