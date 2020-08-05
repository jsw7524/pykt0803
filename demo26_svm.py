from sklearn import datasets, svm

from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt

# iris = datasets.load_iris()
# pca = PCA(n_components=2)
# data = pca.fit(iris.data).transform(iris.data)
# print(iris.data.shape)
# print(data.shape)
# # kernel = 'rbf|linear|poly|sigmoid'
# svc = svm.SVC(kernel='linear')
# svc.fit(data, iris.target)
# datamax = data.max(axis=0) + 1
# datamin = data.min(axis=0) - 1
# print(datamax, datamin)
# n = 2000
# X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
#                    np.linspace(datamin[1], datamax[1], n))
# print(X.shape, Y.shape)
# Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
# plt.contour(X, Y, Z.reshape(X.shape), colors='r')
# for c, s in zip([0, 1, 2], ['.', '+', '*']):
#     d = data[iris.target == c]
#     plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)
# plt.show()
# # sigmoid = 0.86
# # rbf = 0.96
# # linear = 0.96666
# print(svc.score(data, iris.target))

X, Y = np.meshgrid(np.linspace(-100,100, 11),
                    np.linspace(-100, 100, 11))
plt.contour(X, Y, X**2+Y**2, colors='b')
plt.show()