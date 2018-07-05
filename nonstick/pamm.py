import numpy as np

N = 900


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
    pass


def quick_shift():
    pass


def build_gmm():
    pass
