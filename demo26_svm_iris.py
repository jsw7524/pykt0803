from sklearn import datasets, svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
pca = PCA(n_components=2)
data = pca.fit(iris.data).transform(iris.data)
print(iris.data.shape)
print(data.shape)

for c, s in zip([0, 1, 2], ['.', '+', '*']):
    d = data[iris.target == c]
    plt.scatter(d[:, 0], d[:, 1], c='k', marker=s)
plt.show()