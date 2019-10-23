from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

centers = [[-2, 2], [2, 2], [0, 4]]
X, y = make_blobs(n_samples=60, centers=centers, random_state=0, cluster_std=0.60)

# print(y)
# x1 = X[:,0]
# y1 = X[:,1]
# print(X)
# print(x1)
# plt.scatter(x1, y1, color='blue')
# plt.show()

c = np.array(centers)
plt.figure(figsize=(16, 10), dpi=144)
plt.scatter(X[:,0], X[:,1], c=y, s=30)
plt.scatter(c[:,0], c[:,1], c='orange', s=30, marker='^')

k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)

X_sample = [[0, 2]]
y_sample = clf.predict(X_sample)
print(type(y_sample))
neighbors = clf.kneighbors(X_sample, return_distance=False)
plt.scatter(X_sample[0][0], X_sample[0][1], marker='x', c='b', s=100)
print(type(neighbors))
print(neighbors)
# print(y_sample)
for i in neighbors[0]:
	plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]], 'k--', linewidth=0.6)


plt.show()   