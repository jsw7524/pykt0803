import matplotlib.pyplot as plt
from sklearn import linear_model

features = [[0, 1], [1, 3], [2, 8]]
values = [1, 4, 5.5]
regressior1 = linear_model.LinearRegression()
regressior1.fit(features, values)

print('coef=', regressior1.coef_)
print('intercept', regressior1.intercept_)
print(regressior1.coef_[0], regressior1.coef_[1])
newX = [[0.8, 0.8], [2, 1], [10, 14]]
guessY = regressior1.predict(newX)
print(f'predict as {guessY}')
r_score = regressior1.score(newX, guessY)
print(f"calculate r score = {r_score}")
r_score2 = regressior1.score(newX, [4, 9, 35])
print(f"calculate r score again ={r_score2}")

