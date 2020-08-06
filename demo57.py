import numpy
from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

dataset1 = numpy.loadtxt('data/demo53_datasets.csv', delimiter=',', skiprows=1)
print(type(dataset1), dataset1.shape)

inputList = dataset1[:, 0:8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)


def createModel(layer1Count=8, layer2Count=10,
                optimizer='adam', init='uniform'):
    # global model
    model = Sequential()
    print(type(model))
    model.add(Dense(layer1Count, input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(layer2Count, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model


model1 = KerasClassifier(build_fn=createModel, verbose=0)
optimizers = ['rmsprop', 'adam', 'sgd']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
layer1Counts = [8, 10, 12, 14]
layer2Counts = [10, 20, 30, 40]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches, init=inits,
                  layer1Count=layer1Counts, layer2Count=layer2Counts)
grid = GridSearchCV(estimator=model1, param_grid=param_grid)
grid_result = grid.fit(inputList, resultList)

print("Best %f using %s" % (grid_result.best_score_, grid_result.best_params_))


grid = GridSearchCV(estimator=model1, param_grid=param_grid, cv=3)