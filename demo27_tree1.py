from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
classifier1 = tree.DecisionTreeClassifier()
classifier1.fit(X, Y)
print(classifier1)

print(classifier1.predict([[2, 2], [0, 2], [2, 0], [-2, -2]]))