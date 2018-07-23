import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix


def main():
    N = 900
    np.random.seed(7)

    # read example data
    data = pd.read_csv('examples/example_data.csv', index_col=0)
    X = data.values[:, :2]
    y = data.values[:, 2]
    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # get sample grid for KDE with farthest point algorithm
    M = np.sqrt(N)
    y = farthest_point_grid(X, M)

    P = density_estimation(X, y)

    plt.scatter(X[:, 0], X[:, 1], c='k', alpha=0.3, s=10)
    plt.scatter(y[:, 0], y[:, 1], c='b', s=(P * 50) ** 2)

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

    sigma = np.zeros(y_dists.shape[0])
    pdf = np.zeros(y_dists.shape[0])

    for idx in np.arange(0, y_dists.shape[0]):
        not_idx = np.isin(np.arange(0, y_dists.shape[0]), idx, invert=True)
        sigma[idx] = np.amin(y_dists[idx, not_idx])

    x_y_dists = distance_matrix(x, y)
    min_dists = np.amin(x_y_dists, axis=1)

    for idx in np.arange(0, y_dists.shape[0]):
        this_y = x_y_dists[:, idx]
        close_points = np.nonzero(this_y[np.isin(this_y, min_dists)])
        prefactor = np.power(2 * np.pi * np.power(sigma[idx], 2.), (- D / 2.))
        gaussians = prefactor * (np.exp(-np.power(close_points, 2.)
                                 / (2 * np.power(sigma[idx], 2.))))
        pdf[idx] = np.sum(gaussians)

    return pdf


def quick_shift():
    pass


def build_gmm():
    pass




if __name__ == '__main__':
    main()
