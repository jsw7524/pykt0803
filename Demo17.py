
from sklearn import linear_model, datasets
import numpy as np
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
print(type(diabetes), diabetes.data.shape, diabetes.target.shape)
dataForTest = -50
data_train = diabetes.data[:dataForTest]
print(f"train data shape:{data_train.shape}")
target_train = diabetes.target[:dataForTest]
print(f"target data shape:{target_train.shape}")
data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]
#
regression1 = linear_model.LinearRegression(normalize=True)
regression1.fit(data_train, target_train)
print(regression1.coef_)
print(regression1.intercept_)
print('score:', regression1.score(data_test, target_test))
for i in range(dataForTest,0):
    dataArray = np.array(data_test[i]).reshape(1,-1)
    print(f"predict:{regression1.predict(dataArray)[0]:.1f}/actual:{target_test[i]}")