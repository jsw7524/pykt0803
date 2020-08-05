import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans

X = np.r_[np.random.randn(500, 2) + [2, 2],
          np.random.randn(500, 2) + [0, 2],
          np.random.randn(500, 2) + [-2, 2]]

kmeans = KMeans(n_clusters=5, n_init=10, max_iter=30, verbose=1)
kmeans.fit(X)
#print(kmeans.cluster_centers_)
print(kmeans.inertia_)

colors = ['c', 'm', 'y', 'k', 'r', 'g', 'b']
markers = ['o', 'v', '*', 'x', '.', '^', 's']
for i in range(5):
    dataX = X[kmeans.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1],
                c=colors[i], marker=markers[i])
    print(dataX.size)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*',
            s=200, c='#005599')
plt.show()