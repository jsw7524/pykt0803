# import matplotlib.pyplot as plt
# from sklearn  import linear_model
import sklearn.linear_model
regressior1 = sklearn.linear_model.LinearRegression()

features = [[1], [2], [3]]
values = [1, 4, 15]

plt.scatter(features, values, c='green')
plt.show()
