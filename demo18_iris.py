import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

labels = ['sepal length', 'sepal width', 'petal length', 'petal width']
X = iris.data
species = iris.target

counter = 1
for i in range(0, 4):
    for j in range(i + 1, 4):
        plt.figure(counter, figsize=(8, 6))
        counter += 1
        xData = X[:, i]
        yData = X[:, j]
        #plt.clf()
        plt.scatter(xData, yData, c=species, cmap=plt.cm.Paired)
        plt.xlabel(labels[i])
        plt.ylabel(labels[j])
        #plt.xticks([])
        #plt.yticks([])
        plt.show()