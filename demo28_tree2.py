from subprocess import check_call

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz

X = [[0, 0], [1, 1], [0, 1], [1, 0]]
Y = [0, 0, 1, 1]
col = ['red', 'green']
marker = ['o', 'd']
index = 0
while index < len(X):
    type = Y[index]
    plt.scatter(X[index][0], X[index][1], c=col[type],
                marker=marker[type])
    index += 1
plt.show()

classifier1 = tree.DecisionTreeClassifier()
classifier1.fit(X, Y)
export_graphviz(classifier1, out_file='demo28.dot',
                filled=True, rounded=True, special_characters=True)
check_call(['dot', '-Tsvg', 'demo28.dot', '-o', 'demo28.svg'])
#check_call(['dot', '-Tpdf', 'demo28.dot', '-o', 'demo28.pdf'])
#check_call(['dot', '-Tpng', 'demo28.dot', '-o', 'demo28.png'])