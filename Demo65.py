import numpy as np
from keras.datasets import imdb
from keras import models, layers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])
print(max([max(s) for s in train_data]))
word_index = imdb.get_word_index()
print(type(word_index))
# print(word_index)
reverse_word_index = dict([(v, k) for k, v in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[2]])
print(decoded_review)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(train_data[2])
print(x_train[2])

model1 = models.Sequential()
model1.add(layers.Dense(32, activation='relu', input_shape=(10000,)))
model1.add(layers.Dense(16, activation='relu'))
model1.add(layers.Dense(1, activation='sigmoid'))
print(model1.summary())
model1.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model1.fit(x_train, y_train, epochs=30, batch_size=512, validation_data=(x_test, y_test))