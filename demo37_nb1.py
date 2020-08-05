import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
classifier1 = GaussianNB()
classifier1.fit(X, Y)
NewX = [[1, 0], [0, 1], [-1, 0], [0, -1]]
print(classifier1.predict(NewX))

classifier2 = GaussianNB()
classifier2.partial_fit(X, Y, np.unique(Y))
print("predict1", classifier2.predict(NewX))
classifier2.partial_fit([[0.5, 0]], [1])
print("predict2", classifier2.predict(NewX))