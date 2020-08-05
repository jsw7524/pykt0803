import matplotlib.pyplot as plt
from sklearn import linear_model
#
features = [[0, 1], [1, 3], [2, 8]]
values = [1, 4, 5.5]
regressior1 = linear_model.LinearRegression()
regressior1.fit(features, values)

print('coef=',regressior1.coef_)
print('intercept',regressior1.intercept_)
print(regressior1.coef_[0], regressior1.coef_[1])
print(regressior1.coef_[0]*features[0][0]+regressior1.coef_[1]*features[0][1] + regressior1.intercept_)

