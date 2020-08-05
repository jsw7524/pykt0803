
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import numpy as np

import matplotlib.pyplot as plt

iris = datasets.load_iris()

print(list(iris.keys()))
print(iris["feature_names"])
print(iris["target_names"])
X = iris["data"][:, 3:]
y = (iris["target"] == 2).astype(np.int)
regression1 = LogisticRegression()
regression1.fit(X, y)
print(regression1.coef_, regression1.intercept_)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = regression1.predict_proba(X_new)

plt.plot(X, y, "gs")
plt.plot(X_new, y_prob[:, 1], "r-", label="is iris-verginica")
plt.plot(X_new, y_prob[:, 0], "b--", label="is NOT iris-verginica")
plt.xlabel("petal width (cm)")
plt.ylabel("is verginica probability")
plt.legend()
plt.show()