
import numpy
from keras.layers import Dense
from keras.models import Sequential

dataset1 = numpy.loadtxt('data/demo53_datasets.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)