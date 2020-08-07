import tensorflow as tf
import keras
import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import fashion_mnist
from keras import Sequential, layers

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
               'Bag', 'Ankle boot']


def plotOne(x):
    plt.figure()
    plt.imshow(train_images[x])
    plt.colorbar()
    plt.grid(False)
    plt.show()


# plotOne(1)

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
# plt.show()
print(train_images.shape, train_labels.shape)
print(train_images[0])
print(train_labels[:10])
model1 = Sequential([layers.Flatten(input_shape=(28, 28)),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(10, activation='softmax')])
print(model1.summary())
model1.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
model1.fit(train_images, train_labels, epochs=5)
test_loss, test_accuracy = model1.evaluate(test_images, test_labels, verbose=2)
print("test accuracy:", test_accuracy)

##################################################

predictions = model1.predict(test_images)

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols


def plot_image(i, prediction_array, true_labels, images):
    true_label = true_labels[i]
    image = images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image, cmap=plt.cm.binary)
    prediction_label = np.argmax(prediction_array)
    if prediction_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.colorbar()
    plt.xlabel("hello world", color=color)


plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)

def plot_value_array(i, prediction_array, true_labels):
    true_label = true_labels[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_array, color='#888888')
    plt.ylim(0,1)
    predicted_label=np.argmax(prediction_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)