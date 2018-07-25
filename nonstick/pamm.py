import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix


def main():
    np.random.seed(7)

    # read example data
    data = pd.read_csv('../examples/example_data.csv', index_col=0)
    X = data.values[:, :2]
    y = data.values[:, 2]
    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # get sample grid for KDE with farthest point algorithm
    N = X.shape[0]
    M = np.sqrt(N)
    y = farthest_point_grid(X, M)

    P = density_estimation(X, y)

    plt.scatter(X[:, 0], X[:, 1], c='k', alpha=0.3, s=10)
    plt.scatter(y[:, 0], y[:, 1], c='b', s=(P/5))

    plt.show()
    print('lol')


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def farthest_point_grid(x, m):
    farthest_pts = np.zeros((int(m), 2))
    farthest_pts[0] = x[np.random.randint(len(x))]
    distances = calc_distances(farthest_pts[0], x)
    for i in range(1, int(m)):
        farthest_pts[i] = x[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], x))
    return farthest_pts


def density_estimation(x, y):
    D = x.shape[1]
    y_dists = distance_matrix(y, y)

    y_dists[y_dists == 0] = 1000  # large value to prevent self selection
    delta_i = np.amin(y_dists, axis=1)

    y_x_dists = distance_matrix(y, x)
    min_dists = np.argmin(y_x_dists, axis=0)

    sigma_j = np.array([delta_i[idx] for idx in min_dists])

    prefactor = np.power(2 * np.pi * np.power(sigma_j, 2.), (- D / 2.))
    gaussians = prefactor * (np.exp(-np.power(y_x_dists, 2.)
                             / (2 * np.power(sigma_j, 2.))))

    pdf = np.sum(gaussians, axis=1)

    return pdf


def quick_shift():
    pass


def build_gmm():
    pass


if __name__ == '__main__':
    main()
