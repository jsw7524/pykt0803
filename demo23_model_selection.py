import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
data = iris.data
target = iris.target

logisticRegression1 = LogisticRegression()
score = model_selection.cross_val_score(logisticRegression1, data, target, cv=5)
print(score)