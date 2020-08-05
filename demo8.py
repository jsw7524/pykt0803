import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

regression1 = datasets.make_regression(100, 1, noise=30)
print(type(regression1), len(regression1))
print(type(regression1[0]), type(regression1[1]), regression1[0].shape, regression1[1].shape)
# c='red','#C0FFEE'
plt.scatter(regression1[0], regression1[1], c='#C0FF00', marker='.')
plt.show()
regressior1 = linear_model.LinearRegression()
regressior1.fit(regression1[0], regression1[1])

print(f"coef={regressior1.coef_}, intercept={regressior1.intercept_}")
print(f'score={regressior1.score(regression1[0], regression1[1])}')

range1 = [-3, 3]
plt.plot(range1, regressior1.coef_ * range1 + regressior1.intercept_)
plt.scatter(regression1[0], regression1[1], c='#C0FF00', marker='.')
plt.show()