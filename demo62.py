import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras import Sequential
from keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

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


def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=10, verbose=0)
fiveFold = KFold(n_splits=3, shuffle=True)
results = cross_val_score(estimator, features, dummy_y, cv=fiveFold)
print(f"accuracy:{results.mean()}, std:{results.std()}")