import numpy as np
from matplotlib import pyplot as plt

def G(a, b, c, d):
    return lambda x1, x2: np.exp(-(a*(x1-b)**2 + c*(x2-d)**2))

def dG_dx1(a, b, c, d):
    return lambda x1, x2: np.exp(-(a*(x1-b)**2 + c*(x2-d)**2))*(-2.)*a*(x1-b)

def dG_dx2(a, b, c, d):
    return lambda x1, x2: np.exp(-(a*(x1-b)**2 + c*(x2-d)**2))*(-2.)*c*(x2-d)

F = lambda x1, x2: \
    G(10., .2, 3., .4)(x1, x2) + G(6., .8, 12., .6)(x1, x2) + 1.5*(x2-.3)**3

JF = lambda x1, x2: (
    dG_dx1(10., .2, 3., .4)(x1, x2) + dG_dx1(6., .8, 12., .6)(x1, x2),
    dG_dx2(10., .2, 3., .4)(x1, x2) + dG_dx2(6., .8, 12., .6)(x1, x2) + 4.5*(x2-.3)**2)

def draw_gradients(n=20, scale=.02, F=F, JF=JF):
    F_points = np.vectorize(F)(*np.meshgrid(np.linspace(0,1,25), np.linspace(0,1,25)))
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim((0.,1.))
    ax.set_ylim((0.,1.))
    im = ax.imshow(
        F_points, cmap='tab20b', interpolation='bicubic', 
        origin="lower", extent=(0, 1, 0, 1)
    )
    for x1 in np.linspace(0,1,n):
        for x2 in np.linspace(0,1,n):
            d1, d2 = JF(x1, x2)
            d1 *= scale
            d2 *= scale
            ax.arrow(
                x1, x2, d1, d2, width=.001, head_width=0.006,
                head_length=0.012, length_includes_head=True
            )
    fig.colorbar(im, ax=ax)
