from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from scipy.stats import norm as normal


def draw(X, mu_true, tau_true, history):
    n_steps = history.shape[1]
    fig = plt.figure(figsize=(9,2.5))
    ax = fig.add_subplot(132)
    line_loss = ax.plot(np.arange(n_steps), history[0])[0]
    line_loss.set_data([], [])
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax = fig.add_subplot(133)
    ax.scatter(mu_true, tau_true, s=43, facecolors="none", edgecolors="red")
    im_theta = ax.scatter(history[1], history[2], s=5, c=range(n_steps), alpha=.5)
    ax.set_xlabel("mu")
    ax.set_ylabel("tau")
    fig.colorbar(im_theta, ax=ax)
    ax = fig.add_subplot(131)
    ax.hist(X, bins=43, density=True)
    xlim = (X.min(), X.max())
    _x = np.linspace(xlim[0], xlim[1], 100)
    line_density = ax.plot(_x, normal.pdf(_x, loc=history[1,0], scale=np.exp(history[2,0])))[0]
    line_density.set_data([], [])
    fig.tight_layout()

    def frame_processor(j):
        line_loss.set_data(np.arange(j), history[0,:j])
        line_density.set_data(_x, normal.pdf(_x, loc=history[1,j], scale=np.exp(history[2,j])))
        im_theta.set_offsets(history[1:,:j].T)
        return line_loss, line_density, im_theta

    ani = animation.FuncAnimation(
        fig,
        frame_processor,
        range(history.shape[1]),
        blit=True,
        interval=10, # ms
        repeat=False)
    return ani
