from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture as GM
from scipy.stats import norm as normal
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neighbors import KNeighborsRegressor as KNNR

def data_generative(loc, scale, n_samples, rng):
    return rng.normal(loc=loc, scale=scale, size=n_samples)

def data_classification(loc1, scale1, loc2, scale2, n_samples, rng):
    X = np.concatenate((
        rng.normal(loc=loc1, scale=scale1, size=n_samples),
        rng.normal(loc=loc2, scale=scale2, size=n_samples)
    ))
    Y = np.concatenate((
        np.zeros(n_samples, dtype=np.int),
        np.ones(n_samples, dtype=np.int)
    ))
    return X, Y

def data_regression(x_loc, x_scale, noise_scale, a, b, n_samples, rng):
    X = rng.normal(loc=x_loc, scale=x_scale, size=n_samples)
    noise = rng.normal(loc=0., scale=noise_scale, size=n_samples)
    Y = a * X + b + noise
    return X, Y


def draw_generative(loc, scale, n_samples, n_components):
    rng = np.random.RandomState(seed=43)
    X_tr = data_generative(loc, scale, n_samples, rng)
    X_te = data_generative(loc, scale, n_samples, rng)
    X = np.concatenate((X_tr, X_te))
    gm = GM(n_components=n_components)
    gm.fit(X_tr.reshape(-1,1))
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    _x = np.linspace(X.min(), X.max(), 437)
    _true_density = normal.pdf(_x, loc=loc, scale=scale)
    _est_density = np.exp(gm.score_samples(_x.reshape(-1,1)))
    _max = _true_density.max()
    ax.plot(_x, _true_density, label="true density")
    ax.plot(_x, _est_density, label="estimated density")
    ax.scatter(X_te, np.zeros(n_samples)-0.043*_max, marker="|", s=143, label="test set")
    ax.scatter(X_tr, np.zeros(n_samples)+0.043*_max, marker="|", s=143, label="training set")
    ax.legend()
    ax.set_ylim((-.1*_max, 2.*_max))
    print("negative mean log likelihood (less is better)")
    print("true train: {:.3f}\ntrue test: {:.3f}\nest train: {:.3f}\nest test: {:.3f}".format(
        - normal.logpdf(X_tr, loc=loc, scale=scale).mean(),
        - normal.logpdf(X_te, loc=loc, scale=scale).mean(),
        - gm.score_samples(X_tr.reshape(-1,1)).mean(),
        - gm.score_samples(X_te.reshape(-1,1)).mean()
    ))

def draw_regression(x_loc, x_scale, a, b, n_samples, n_neighbors=20, weights="distance"):
    rng = np.random.RandomState(seed=43)
    X_tr, Y_tr = data_regression(x_loc, x_scale, 1., a, b, n_samples, rng)
    X_te, Y_te = data_regression(x_loc, x_scale, 1., a, b, n_samples, rng)
    X = np.concatenate((X_tr, X_te))
    knn = KNNR(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_tr.reshape(-1,1), Y_tr)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    _x = np.linspace(X.min(), X.max(), 100*n_samples)
    ax.plot(_x, a*_x+b, label="true expected y")
    ax.plot(_x, knn.predict(_x.reshape(-1,1)), label="estimated expected y")
    ax.scatter(X_te, Y_te, label="test set")
    ax.scatter(X_tr, Y_tr, label="training set")
    ax.legend()
    print("mean squared error (less is better)")
    print("train: {:.3f}\ntest: {:.3f}".format(
        np.mean(np.square(knn.predict(X_tr.reshape(-1,1)) - Y_tr)),
        np.mean(np.square(knn.predict(X_te.reshape(-1,1)) - Y_te))
    ))

def draw_classification(loc0, scale0, loc1, scale1, n_samples, n_neighbors=20, weights="distance", epsilon=1e-6):
    rng = np.random.RandomState(seed=43)
    X_tr, Y_tr = data_classification(loc0, scale0, loc1, scale1, n_samples, rng)
    X_te, Y_te = data_classification(loc0, scale0, loc1, scale1, n_samples, rng)
    X = np.concatenate((X_tr, X_te))
    knn = KNN(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_tr.reshape(-1,1), Y_tr)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    _x = np.linspace(X.min(), X.max(), 437)

    def _proba(X):
        _pdf0 = normal.pdf(X, loc=loc0, scale=scale0)
        _pdf1 = normal.pdf(X, loc=loc1, scale=scale1)
        _0_proba = _pdf0 / (_pdf0 + _pdf1)
        _1_proba = _pdf1 / (_pdf0 + _pdf1)
        return np.concatenate((
            _0_proba.reshape(-1,1),
            _1_proba.reshape(-1,1)
        ), axis=1)

    _true_1_proba = _proba(_x)[:,1]
    _est_1_proba = knn.predict_proba(_x.reshape(-1,1))[:,1]

    ax.plot(_x, _true_1_proba, label="true class 1 probability", c="#FF0000")
    ax.plot(_x, _est_1_proba, label="estimated class 1 probability", c="#0000FF")

    ax.scatter(X_tr[Y_tr==0], np.zeros(n_samples)-0.021, marker="|", s=143, label="training set class 0", c="#008888")
    ax.scatter(X_tr[Y_tr==1], np.ones(n_samples)+0.021, marker="|", s=143, label="training set class 1", c="#0000DD")
    ax.scatter(X_te[Y_te==0], np.zeros(n_samples), marker="|", s=143, label="test set class 0", c="#888800")
    ax.scatter(X_te[Y_te==1], np.ones(n_samples), marker="|", s=143, label="test set class 1", c="#DD0000")

    ax.legend()

    def _loss(X_proba, Y):
        return - np.mean(np.log(np.clip(X_proba[range(X_proba.shape[0]),Y], a_min=epsilon, a_max=1-epsilon)))

    print("negative mean log likelihood (less is better)")
    print("true train: {:.3f}\ntrue test: {:.3f}\nest train: {:.3f}\nest test: {:.3f}".format(
        _loss(_proba(X_tr), Y_tr),
        _loss(_proba(X_te), Y_te),
        _loss(knn.predict_proba(X_tr.reshape(-1,1)), Y_tr),
        _loss(knn.predict_proba(X_te.reshape(-1,1)), Y_te)
    ))
    print("accuracy (more is better)")
    print("train: {:.3f}\ntest: {:.3f}".format(
        (knn.predict(X_tr.reshape(-1,1)) == Y_tr).mean(),
        (knn.predict(X_te.reshape(-1,1)) == Y_te).mean()
    ))
