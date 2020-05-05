import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
print(iris)
# print("Input: ", iris.data)  # or feature points
# print("Output: ", iris.target)  # or target points or labels
# print("Features: ", iris.feature_names)
# print("Labels: ", iris.target_names)
df = pd.DataFrame({'x': iris.data[:, 0],  # all rows  0th column
                   'y': iris.data[:, 1],  # all rows  1th column
                   'cluster': iris.target})
print(df)

# now we will calculate the centroids:
centroids = {}
for i in range(3):
    result_list = []
    result_list.append(df.loc[df['cluster'] == i]['x'].mean())  # pick the value x where clluster = 0,1,2
    result_list.append(df.loc[df['cluster'] == i]['y'].mean())  # pick the value y where clluster = 0,1,2
    centroids[i] = result_list
print(centroids)

# here we will plot features and target value on the graph
# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], c=iris.target, cmap='gist_rainbow')
# plt.xlabel("sepal length")
# plt.ylabel("sepal width")
# plt.show()


# colmap = {0: 'r', 1: 'g', 2: 'b'}
# #  centroids ===> {0: [5.006, 3.428], 1: [5.936, 2.7700000000000005], 2: [6.587999999999998, 2.974]}
# for i in range(3):
#     # centroids is calculated by taking its value centroids[i][0] it means centroid's 0 key value at 0 i.e 5.00599
#     # centroids[i][1] it means centroid's 0 key value at 3.42800
#     plt.scatter(centroids[i][0], centroids[i][1], color=colmap[i])
# plt.show()


# now we will combine our feature,targets or labels and centroids
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], c=iris.target, cmap='gist_rainbow', alpha=0.3)
colmap = {0: 'r', 1: 'g', 2: 'b'}
for i in range(3):
    # centroids is calculated by taking its value centroids[i][0] it means centroid's 0 key value at 0 i.e 5.00599
    # centroids[i][1] it means centroid's 0 key value at 3.42800
    plt.scatter(centroids[i][0], centroids[i][1], color=colmap[i], edgecolors="k")
plt.show()


data = scale(iris.data)  # this is our features and we scale it down B/W 1 to -1
# bcoz our features might contain very larege values that could not be easily computed
y = iris.target  # labels

k = 10  # clusters or centroids
samples, features = data.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (
        name, estimator.inertia_, metrics.homogeneity_score(y, estimator.labels_),
        metrics.completeness_score(y, estimator.labels_), metrics.v_measure_score(y, estimator.labels_),
        metrics.adjusted_rand_score(y, estimator.labels_), metrics.adjusted_mutual_info_score(y, estimator.labels_),
        metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))


clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)

