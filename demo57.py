import numpy
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import StratifiedKFold

dataset1 = numpy.loadtxt('data/demo53_datasets.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)

fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
totalScores = []


def createModel():
    model = Sequential()
    print(type(model))
    model.add(Dense(8, input_dim=8, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model



for train, test in fiveFold.split(inputList, resultList):
    # print(type(train))
    # print(type(test))
    # print(train)
    # print(test)

    model1 = createModel()
    model1.fit(inputList[train], resultList[train], epochs=200, batch_size=20, verbose=0)
    scores = model1.evaluate(inputList[test], resultList[test])
    print(scores)
    totalScores.append(scores[1] * 100)
    print(model1.metrics_names)
    print(f"{model1.metrics_names[0]} => {scores[0]}")
    print(f"{model1.metrics_names[1]} => {scores[1]}")

print(f"five fold scores={totalScores}")