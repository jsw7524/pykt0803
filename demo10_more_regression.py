demo10_more_regression

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
x, y = make_regression(n_samples=1000, n_features=10, n_informative=2)
model = LinearRegression()
model.fit(x, y)
importance = model.coef_
print(importance)
for i, v in enumerate(importance):
    print(f"feature:{i:0d}, score:{v:.3f}")
pyplot.bar([x for x in range(len(importance))],importance)
pyplot.show()