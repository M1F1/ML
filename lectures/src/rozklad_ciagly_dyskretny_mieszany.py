from collections import Counter

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm as normal
from scipy.stats import multivariate_normal

from IPython.display import HTML as html_print

def draw_discrete(probability, n_samples):
    p = probability
    rng = np.random.RandomState(seed=43)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(2, 4, 1)
    ax.bar(list(range(len(p))), p)
    ax.set_title("true distribution")
    def draw(n_samples, which_subplot):
        samples = np.array([rng.choice(range(len(p)), p=p) for _ in range(n_samples)])
        c_samples = [Counter(samples)[i] for i in range(len(p))]
        ax = fig.add_subplot(2, 4, which_subplot)
        ax.bar(list(range(len(p))), c_samples)
        ax.set_title("{} samples".format(n_samples))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for i, n in enumerate(n_samples[:4]):
        draw(n_samples=n, which_subplot=5+i)
    fig.tight_layout()

def draw_normal(loc, scale, n_samples):
    rng = np.random.RandomState(seed=43)
    fig = plt.figure(figsize=(8,3))
    ax = fig.add_subplot(111)
    y = rng.normal(size=n_samples, loc=loc, scale=scale)
    x = np.linspace(min(-5, y.min()), max(5, y.max()),437)
    ax.plot(x, normal.pdf(x, loc, scale))
    ax.scatter(y, 0*y, alpha=.3, s=143)
    fig.tight_layout()

def draw_normal_2d(mean, cov, n_samples):
    rng = np.random.RandomState(seed=43)
    rx, ry = rng.multivariate_normal(size=n_samples, mean=mean, cov=cov).T
    xlim = (min(rx.min(),-3+mean[0]), max(rx.max(),3+mean[0]))
    ylim = (min(ry.min(),-3+mean[1]),max(ry.max(),3+mean[1]))
    x, y = np.meshgrid(
        np.linspace(*xlim,30),
        np.linspace(*ylim,30)
    )
    z = multivariate_normal.pdf(
            np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1),
            mean=mean,
            cov=cov
    ).reshape(x.shape)
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111)
    im = ax.contourf(x, y, z, levels=43)
    fig.colorbar(im, ax=ax)
    ax.scatter(rx, ry, alpha=.3, s=43, c="white")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.tight_layout()

def sample_2d(n_samples):
    # https://github.com/jupyter/notebook/issues/2284
    def cstr(s, color):
        return "<text style=color:{}>{}</text>".format(color, s)
    rng = np.random.RandomState(seed=43)
    def sample():
        THRESHOLD1 = 0.
        THRESHOLD2 = 0.
        return max(THRESHOLD1, rng.normal()), max(THRESHOLD2, rng.normal())
    for _ in range(n_samples):
        x1, x2 = sample()
        if x1 == 0. and x2 == 0:
            color = "green"
        elif x1 > 0. and x2 > 0.:
            color = "black"
        elif x1 > 0.: # and x2 == 0
            color = "blue"
        else: # x1 == 0 and x2 > 0
            color = "red"
        s = "(x1, x2): (%.3f, %.3f)" % (x1, x2)
        display(html_print(cstr(s, color)))