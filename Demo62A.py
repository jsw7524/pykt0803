import numpy
from keras.datasets import imdb
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = imdb.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
X = numpy.concatenate((X_train, X_test), axis=0)
Y = numpy.concatenate((y_train, y_test), axis=0)
print(X[0])
print(X.shape)
print(Y.shape)
print(f"y category={numpy.unique(Y)}")
print(f"X={len(numpy.unique(numpy.hstack(X)))}")
result = [len(x) for x in X]
print(f"mean={numpy.mean(result)}, std={numpy.std(result)}")
plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result)
plt.show()