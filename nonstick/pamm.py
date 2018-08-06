import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix


def main():
    np.random.seed(7)
    data_dir = op.join(op.dirname(__file__), '../examples')
    data_file = op.join(data_dir, 'example_data.csv')
    # read example data
    data = pd.read_csv(data_file, index_col=0)
    X = data.values[:, :2]

    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # get sample grid for KDE with farthest point algorithm
    N = X.shape[0]
    M = np.sqrt(N).round()
    y = farthest_point_grid(X, M)

    P = density_estimation(X, y)
    clust = quick_shift(y, P)
    gmm = build_gmm(y, P, clust)
    print('number of clusters: ', len(np.unique(clust)))

    plt.scatter(X[:, 0], X[:, 1], c='k', alpha=0.3, s=10)
    plt.scatter(y[:, 0], y[:, 1], c=clust, s=(P/5))

    plt.show()


def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def farthest_point_grid(x, m):
    farthest_pts = np.zeros((int(m), 2))
    farthest_pts[0] = x[np.random.randint(len(x))]
    distances = calc_distances(farthest_pts[0], x)
    for i in range(1, int(m)):
        farthest_pts[i] = x[np.argmax(distances)]
        distances = np.minimum(distances,
                               calc_distances(farthest_pts[i], x))
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


def quick_shift(y, P):
    # get y distances
    y_dists = distance_matrix(y, y)
    y_dists[y_dists == 0] = 1000
    lamb = y_dists.min(axis=0).mean() * 1.3

    # create cluster id array to assign clusters
    clusters = np.zeros_like(P, dtype='int')

    def connect_to_neighbor(i):
        # find points with greater probability
        mask = np.where(P > P[i])[0]
        # return if we hit the highest probability point
        if len(mask) == 0:
            return i

        # get the id of the closest higher probability point
        min_dist_id = np.argmin(y_dists[mask, i])
        j = mask[min_dist_id]
        if y_dists[i, j] > lamb:
            return i

        return connect_to_neighbor(j)

    # for each y, climb to highest prob within lambda
    for idx in range(0, len(P)):
        clusters[idx] = connect_to_neighbor(idx)

    return clusters


def build_gmm(y, P, clusters):
    total_P = P.sum()
    # get position of each cluster center
    z_k = y[np.unique(clusters)].copy()
    # get pk for cluster
    p_k = np.array([P[np.where(clusters == clust_id)].sum()/P.sum()
                    for clust_id in np.unique(clusters)])
    # get sigma for each cluster
    sigma_k = np.zeros(shape=(len(z_k), y.shape[1], y.shape[1]))
    for idx, clust_id in enumerate(np.unique(clusters)):
        members = np.where(clusters == clust_id)
        distances = y[members] - z_k[idx]
        probs = P[members] / total_P
        sigma_k[idx] = np.cov(distances.T,
                              aweights=(P[members] / total_P))

    # construct and return models? maybe check sklearn for format...
    return


if __name__ == '__main__':
    main()
