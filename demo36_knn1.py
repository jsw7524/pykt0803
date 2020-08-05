import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# manual make a data directory, and download sonar.all-data
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
df = pd.read_csv(URL, header=None, prefix="X")
# df = pd.read_csv(URL, header=None, prefix="X")
print(df.shape)
print(df.columns)
#
df.rename(columns={'X60': 'Label'}, inplace=True)
# n_neighbots=2,4
clf1 = KNeighborsClassifier(n_neighbors=6)
data, labels = df.iloc[:, :-1], df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
clf1.fit(X_train, y_train)
y_predict = clf1.predict(X_test)
print("score=", clf1.score(X_test, y_test))
# get confusion matrix
result_cm1 = confusion_matrix(y_test, y_predict)
print(result_cm1)

scores = cross_val_score(clf1, data, labels, cv=5, groups=labels)
print(scores)
from joblib import dump, load

dump(clf1, "knn1.joblib")
knn2 = load("knn1.joblib")
y_predict2 = knn2.predict(X_test)
result2 = confusion_matrix(y_predict, y_predict2)
print(result2)