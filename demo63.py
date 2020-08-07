import numpy as np
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])
print(max([max(s) for s in train_data]))
word_index = imdb.get_word_index()
print(type(word_index))
print(word_index)
reverse_word_index = dict([(v, k) for k, v in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[1]])
print(decoded_review)