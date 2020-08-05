import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn import datasets

iris = datasets.load_iris()

df1 = pd.DataFrame(iris.data, columns=iris.feature_names)
df1['species'] = np.array([iris.target_names[i] for i in iris.target])
print(df1['species'])
seaborn.pairplot(df1, hue='species')
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test =train_test_split(df1[iris.feature_names], iris.target,
                                                  test_size=0.5, stratify=iris.target)
from sklearn.ensemble import RandomForestClassifier
rf1 = RandomForestClassifier(n_estimators=100, oob_score=True)
rf1.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
predicted = rf1.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
print(f'oob estimator score={rf1.oob_score_}')
print(f"accuracy={accuracy}")

from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns=iris.target_names, index=iris.target_names)
seaborn.heatmap(cm, annot=True)
plt.show()