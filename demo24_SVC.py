import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm
# get iris data
iris = datasets.load_iris()
data = iris.data
target = iris.target
# get classifier
logisticRegression1 = LogisticRegression()
svc1 = svm.SVC()

classifiers = [logisticRegression1, svc1]
for c in classifiers:
    print(f"---now training using {c.__class__.__name__}---")
    score = model_selection.cross_val_score(c, data, target, cv=5)
    print(score)