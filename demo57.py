import numpy
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

dataset1 = numpy.loadtxt('data/demo53_datasets.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)


def createModel():
    # global model
    model = Sequential()
    print(type(model))
    model.add(Dense(8, input_dim=8, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


model1 = KerasClassifier(build_fn=createModel, epochs=200, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
results = cross_val_score(model1, inputList, resultList, cv=fiveFold)
print(results)
print(f"mean={results.mean()},std={results.std()}")