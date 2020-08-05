import matplotlib.pyplot as plt
from sklearn import linear_model

regressior1 = linear_model.LinearRegression()

features = [[1], [2], [3], [4],[5]]
values = [1, 4, 15, 8, 25]

plt.scatter(features, values, c='green')
regressior1.fit(features, values)
print(regressior1.coef_)
print(regressior1.intercept_)
range1 = [-1, 10]
plt.plot(range1, regressior1.coef_ * range1 + regressior1.intercept_, c='red')
plt.show()
