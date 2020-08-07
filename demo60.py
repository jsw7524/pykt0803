import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
dataFrame1 = pd.read_csv('data/iris.data', header=None)
dataset = dataFrame1.values
print(type(dataFrame1), type(dataset))
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
print(features)
print(labels)
print(dataset.shape)

encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
print(encoded_Y)
dummy_y = np_utils.to_categorical(encoded_Y)
print(type(dummy_y), dummy_y[:50])