import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from scipy.stats import multivariate_normal
import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform


def main():
    np.random.seed(7)
    data_dir = op.join(op.dirname(__file__), "../examples")
    data_file = op.join(data_dir, "example_data.csv")
    # read example data
    data = pd.read_csv(data_file, index_col=0)
    X = data.values[:, :3]  # include class as third dimension
    y = data.values[:, 2]

    # scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # get sample grid for KDE with farthest point algorithm
    N = X.shape[0]
    M = np.sqrt(N).round()
    Y = farthest_point_grid(X_train, M)
    P = density_estimation(X_train, Y)
    clust = quick_shift(Y, P)
    unique_clusts = np.unique(clust)
    # predict with gmm
    agg = agglomerate(X_train, Y)

    agg_clust_id = np.zeros_like(clust)
    for idx, val in enumerate(clust):
        agg_clust_id[idx] = agg[np.where(unique_clusts == val)[0]]

    print("number of clusters: ", len(np.unique(clust)))
    plt.scatter(X_train[:, 0], X_train[:, 1], c="k", alpha=0.3, s=10)

    plt.scatter(Y[:, 0], Y[:, 1], c=agg_clust_id, s=(P / 5))

    plt.show()

    # predict with gmm
    gmm = build_gmm(Y, P, clust)
    best = gmm.predict(X_test)

    plt.scatter(X_train[:, 0], X_train[:, 1], c="k", alpha=0.3, s=10)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=best, s=20)
    plt.colorbar()
    plt.show()


def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def farthest_point_grid(x, m):
    farthest_pts = np.zeros((int(m), x.shape[1]))
    # select random point as first grid point
    farthest_pts[0] = x[np.random.randint(len(x))]
    distances = calc_distances(farthest_pts[0], x)
    # iteratively select farthest remaining point
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

    prefactor = np.power(2 * np.pi * np.power(sigma_j, 2.0), (-D / 2.0))
    gaussians = prefactor * (
        np.exp(-np.power(y_x_dists, 2.0) / (2 * np.power(sigma_j, 2.0)))
    )

    pdf = np.sum(gaussians, axis=1)

    return pdf


def quick_shift(y, P, scaling_factor=2.0):
    # get y distances
    y_dists = distance_matrix(y, y)
    y_dists[y_dists == 0] = 1000
    lamb = y_dists.min(axis=0).mean() * scaling_factor

    # create cluster id array to assign clusters
    clusters = np.zeros_like(P, dtype="int")

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

    # combine single-member clusters w/ nearest neighbor
    for idx, clust_id in enumerate(np.unique(clusters)):
        members = np.where(clusters == clust_id)[0]
        if len(members) == 1:
            yo = np.argmin(y_dists[members[0]])
            clusters[members[0]] = clusters[yo]

    return clusters


def agglomerate(x, y, bootstrap_attempts=100):
    # Calculate pdf and clusters for original sample
    p = density_estimation(x, y)
    clusts = quick_shift(y, p)
    unique_clusts = np.unique(clusts)
    # get position of central clusters
    y_k = np.array(
        [
            np.where(clusts == clust_id)[0]
            for clust_id in unique_clusts
        ]
    )
    Q_k = np.array(
        [
            p[y_k[idx]].sum()
            for idx in range(len(unique_clusts))
        ]
    )
    # number of clusters in original sample...
    num_clusts = Q_k.shape[0]
    a_ij = np.zeros([num_clusts, num_clusts])

    # for each bootstrapped sample... get kde, and
    for m in range(bootstrap_attempts):
        # get bootstrapped sample (sample id m)
        x_m = resample(x, n_samples=1000)
        # calculate pdf and clusters
        p_m = density_estimation(x_m, y)
        clusts_m = quick_shift(y, p_m)

        unique_clusts_m = np.unique(clusts_m)
        for k, clust_id in enumerate(unique_clusts_m):
            # get pk for cluster
            y_k_m = np.where(clusts_m == clust_id)[0]
            Q_k_m = p_m[y_k_m].sum()

            for i in range(num_clusts):
                Q_i = p_m[y_k[i]].sum()
                y_ikm = np.intersect1d(y_k[i], y_k_m)
                Q_ikm = p_m[y_ikm].sum() / Q_k_m

                for j in range(num_clusts):
                    if j >= i:
                        Q_j = p_m[y_k[j]].sum()
                        y_jkm = np.intersect1d(y_k[j], y_k_m)
                        Q_jkm = p_m[y_jkm].sum() / Q_k_m
                        a = (
                            Q_k_m * Q_ikm * Q_jkm
                            / (bootstrap_attempts * np.sqrt(Q_i * Q_j))
                        )
                        a_ij[i, j] += a
                        if i != j:
                            a_ij[j, i] += a

    # identify metaclusters
    assert np.all(a_ij - a_ij.T < 1e-6)
    reduced_distances = squareform(a_ij, checks=False)
    linkage = scipy.cluster.hierarchy.ward(reduced_distances)
    fclust_id = scipy.cluster.hierarchy.fcluster(linkage, t=3, criterion='maxclust')

    return fclust_id


class GaussianMixtureModel:
    def __init__(self, p, z, sigma):
        self.p = p
        self.z = z
        self.sigma = sigma

    def predict(self, X):
        """
        Predict the labels for the data samples in X using trained model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
         List of n_features-dimensional data points. Each row corresponds to a single data point.

        Returns:
        --------
        labels : array, shape (n_samples,)
         Component labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Predict posterior probability of each component given the data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
         List of n_features-dimensional data points. Each row corresponds to a single
         data point.

        Returns:
        --------
        resp : array, shape (n_samples, n_components)
         Returns the probability each Gaussian (state) in the model given each sample.
        """
        probs = np.zeros([X.shape[0], len(self.p)])
        for idx, weight in enumerate(self.p):
            gaussian_prob = multivariate_normal(
                self.z[idx], self.sigma[idx], allow_singular=True
            )
            probs[:, idx] = weight * gaussian_prob.pdf(X)
        return probs

    def score(self, X):
        pass


def build_gmm(y, P, clusters):
    total_P = P.sum()
    # get position of each cluster center
    z_k = y[np.unique(clusters)].copy()
    # get pk for cluster
    p_k = np.array(
        [
            P[np.where(clusters == clust_id)].sum() / P.sum()
            for clust_id in np.unique(clusters)
        ]
    )
    # get sigma for each cluster
    sigma_k = np.zeros(shape=(len(z_k), y.shape[1], y.shape[1]))
    for idx, clust_id in enumerate(np.unique(clusters)):
        members = np.where(clusters == clust_id)
        distances = y[members] - z_k[idx]
        sigma_k[idx] = np.cov(distances.T, aweights=(P[members] / total_P))

    return GaussianMixtureModel(p_k, z_k, sigma_k)


if __name__ == "__main__":
    main()
