from numpy import array
from sklearn.decomposition import PCA

# A = array([[10, 2, 3], [30, 4, 5], [50, 6, 7], [70, 8, 9]])
# print(A)
# # PCA(1), PCA(2), PCA(3)
# pca = PCA(3)
# pca.fit(A)
# print("components",pca.components_)
# print("variance",pca.explained_variance_)
# print("variance ratio:",pca.explained_variance_ratio_)
# B = pca.transform(A)
# print(B)

from numpy import array
from sklearn.decomposition import PCA

A = array([[10, 2, 3], [30, 4, 5], [50, 6, 7], [70, 8, 9]])
print(A)
# PCA(1), PCA(2), PCA(3)
pca = PCA(3)
pca.fit(A)
print("components",pca.components_)
print("variance",pca.explained_variance_)
print("variance ratio:",pca.explained_variance_ratio_)
B = pca.transform(A)
print(B)

