import numpy as np


def toroidalWrapAround_np(points, domain_size=1):  # for arbitrary domain size
    points = np.where(points > 1, points - np.floor(points), points)
    return np.where(points < 0, points + np.ceil(np.abs(points)), points)


def grid_sampler(N, scale_domain=1.0):
    samples = np.zeros((N, 2))
    sqrN = int(np.sqrt(N))
    gridsize = 1.0 / float(sqrN)
    xa = np.linspace(0 + gridsize / 2.0, 1 - gridsize / 2.0, sqrN)
    ya = np.linspace(0 + gridsize / 2.0, 1 - gridsize / 2.0, sqrN)
    xv, yv = np.meshgrid(xa, ya)
    # print(xv.shape, yv.shape)
    r = np.random.rand(2) - 0.5
    samples[:, 0] = xv.reshape(N) + r[0]
    samples[:, 1] = yv.reshape(N) + r[1]
    samples = toroidalWrapAround_np(samples)
    return samples


def jitter_sampler(N, scale_domain=1.0):
    samples = np.zeros((N, 2))
    sqrN = int(np.sqrt(N))
    gridsize = 1.0 / float(sqrN)
    xa = np.linspace(0 + gridsize / 2.0, 1 - gridsize / 2.0, sqrN)
    ya = np.linspace(0 + gridsize / 2.0, 1 - gridsize / 2.0, sqrN)
    xv, yv = np.meshgrid(xa, ya)
    # print(xv.shape, yv.shape)
    r = (np.random.rand(N, 2) - 0.5) * 0.99 / sqrN
    samples[:, 0] = xv.reshape(N) + r[:, 0]
    samples[:, 1] = yv.reshape(N) + r[:, 1]
    return samples
