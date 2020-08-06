import numpy
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

dataset1 = numpy.loadtxt('data/demo53_datasets.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)

feature_train, feature_test, label_train, label_test = \
    train_test_split(inputList, resultList, test_size=0.25, stratify=resultList)

for data in [resultList, label_train, label_test]:
    classes, counts = numpy.unique(data, return_counts=True)
    for cl, co in zip(classes, counts):
        print(f"{int(cl)}==>{co / sum(counts)}")

model = Sequential()
print(type(model))
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(feature_train, label_train, epochs=200, batch_size=20)
scores = model.evaluate(feature_test, label_test)
print(scores)
print(model.metrics_names)
print(f"{model.metrics_names[0]} => {scores[0]}")
print(f"{model.metrics_names[1]} => {scores[1]}")