import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from scipy.spatial import distance_matrix

N = 900


def farthest_point_grid(x, m):
    distances = distance_matrix(x, x)
    distances.shape
    y_id = np.random.randint(len(x), size=1)

    not_y = [i for i in range(len(distances))
             if i not in y_id]
    max_dist = np.amax(distances[y_id, not_y])
    max_loc = np.where(distances[y_id, not_y] == max_dist)[0]
    y_id = np.append(y_id, max_loc)

    for _ in range(1, int(m)):
        not_y = np.array([i for i in range(len(distances))
                          if i not in y_id])
        min_dists = np.amin(distances[y_id][:, not_y], axis=1)
        max_dist = np.amax(min_dists)
        max_loc = np.where(distances[y_id][:, not_y] == max_dist)[1]
        y_id = np.append(y_id, max_loc)

    y_grid = x[y_id, :]
    return y_grid


def density_estimation(x, y):
    pass


def quick_shift():
    pass


def build_gmm():
    pass


### testing ###

import matplotlib.pyplot as plt
# read example data
data = pd.read_csv('../examples/example_data.csv', index_col=0)
X = data.values[:, :2]
y = data.values[:, 2]
# scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# get sample grid for KDE with farthest point algorithm
M = np.sqrt(N)
y = farthest_point_grid(X, M)

plt.scatter(X[:, 0], X[:, 1], c='k', alpha=0.3, s=10)
plt.scatter(y[:, 0], y[:, 1], c='b', s=50)

plt.show()
print('lol')
