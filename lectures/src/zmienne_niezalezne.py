from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import scipy.stats

def draw_indepentent_mixtures(x1_locs, x1_scales, x1_p, x2_locs, x2_scales, x2_p):

    class GaussianMixture:
        def __init__(self, locs, scales, p, seed=43):
            self.locs = locs
            self.scales = scales
            self.p = p
            self.rng = np.random.RandomState(seed=seed)
        def sample(self):
            which = self.rng.choice(range(len(self.p)), p=self.p)
            return self.rng.normal(loc=self.locs[which], scale=self.scales[which])
        def pdf(self, x):
            return float(np.sum(
                [self.p[which] * scipy.stats.norm(loc=self.locs[which], scale=self.scales[which]).pdf(x) \
                    for which in range(len(self.p))]))

    x1_locs, x1_scales, x1_p, x2_locs, x2_scales, x2_p = [
            np.array(x) for x in [x1_locs, x1_scales, x1_p, x2_locs, x2_scales, x2_p]
    ]        
    mg1 = GaussianMixture(
        locs=x1_locs,
        scales=x1_scales,
        p=x1_p
    )
    mg2 = GaussianMixture(
        locs=x2_locs,
        scales=x2_scales,
        p=x2_p
    )
    x1 = np.linspace(x1_locs.min()-3., x1_locs.max()+3., 50)
    x2 = np.linspace(x2_locs.min()-3., x2_locs.max()+3., 50)
    x1v, x2v = np.meshgrid(x1, x2)
    z = np.multiply(np.vectorize(mg1.pdf)(x1v), np.vectorize(mg2.pdf)(x2v))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(x1v, x2v, z)#, cmap="cubehelix")

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121)
    mg1_pdf = np.vectorize(mg1.pdf)(x1)
    ax.plot(x1, mg1_pdf)
    ax = fig.add_subplot(122)
    mg2_pdf = np.vectorize(mg2.pdf)(x2)
    ax.plot(x2, mg2_pdf)
