import sklearn.datasets as datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# get iris data
iris = datasets.load_iris()
data = iris.data
target = iris.target
# get classifier
logisticRegression1 = LogisticRegression()
svc1 = svm.SVC()
tree1 = tree.DecisionTreeClassifier()
rf1 = RandomForestClassifier(n_estimators=100, oob_score=True)

classifiers = [logisticRegression1, svc1, tree1, rf1]
for c in classifiers:
    print(f"---now training using {c.__class__}---")
    score = model_selection.cross_val_score(c, data, target, cv=5)
    print(score)