import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

regression1 = datasets.make_regression(100, 1, noise=10)
print(regression1)
print(type(regression1), len(regression1))
print(type(regression1[0]), type(regression1[1]), regression1[0].shape, regression1[1].shape)
# c='red','#C0FFEE'
plt.scatter(regression1[0], regression1[1], c='red', marker='.')
plt.show()