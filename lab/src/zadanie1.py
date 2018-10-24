import matplotlib
from matplotlib import pyplot as plt
plt.style.use("fivethirtyeight")
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import proj3d
import numpy as np
from numpy.polynomial.polynomial import polyval2d

class ExpPoly2D:

    def __init__(self, cs, ds):
        self.cs = cs
        self.ds = ds
        
    #def __str__(self,):
    #    print(self.cs)
    #    print(self.ds)
        
    def _output(self, x1, x2, c, d):
        return d * np.exp(-np.polynomial.polynomial.polyval2d(x1, x2, c)**2)

    def _d_dx1(self, x1, x2, c, d):
        return \
            d * np.exp(-polyval2d(x1, x2, c)**2) * \
            (-2.) * polyval2d(x1, x2, c) * \
            polyval2d(x1, x2, np.multiply(np.arange(c.shape[0]).reshape(-1,1), c)[1:,:])
            
    def _d_dx2(self, x1, x2, c, d):
        return \
            d * np.exp(-polyval2d(x1, x2, c)**2) * \
            (-2.) * polyval2d(x1, x2, c) * \
            polyval2d(x1, x2, np.multiply(np.arange(c.shape[1]).reshape(1,-1), c)[:,1:])

    def __call__(self, x):
        x1, x2 = x
        return np.sum([self._output(x1, x2, c, d) for c, d in zip(self.cs, self.ds)])
    
    def _gradient(self, x):
        x1, x2 = x
        return np.array([
            np.sum([self._d_dx1(x1, x2, c, d) for c, d in zip(self.cs, self.ds)]),
            np.sum([self._d_dx2(x1, x2, c, d) for c, d in zip(self.cs, self.ds)]),
        ])
    
    def gradient(self, d):
        result = {}
        for k, v in d.items():
            result[k] = np.array([self._gradient(row) for row in v])
        return result


def draw_contours(ax, func, extent, num_linspace=100, num_contours=25):
    X = np.linspace(extent[0], extent[1], num=num_linspace)
    Y = np.linspace(extent[2], extent[3], num=num_linspace)
    Z = np.vectorize(func, signature="(n)->()")(np.stack(np.meshgrid(X, Y), axis=-1))
    cs = ax.contour(X, Y, Z, num_contours, linewidths=1, linestyles="solid")
    ax.clabel(cs, inline=1, fontsize=5)

def draw_surface(func, extent=(-6, 6, -6, 6), num_linspace=60):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    X = np.linspace(extent[0], extent[1], num=num_linspace)
    Y = np.linspace(extent[2], extent[3], num=num_linspace)
    X, Y = np.meshgrid(X, Y)
    Z = np.vectorize(func, signature="(n)->()")(np.stack((X,Y), axis=-1))
    surf = ax.plot_surface(X, Y, Z)

rng_poly = np.random.RandomState(seed=43)
Fun1 = ExpPoly2D(
    cs=[
        .001*(rng_poly.choice(5, size=(3,5))-2),
        .3*rng_poly.normal(size=(3,3)),
        .8*rng_poly.normal(size=(6,6)),
        .01*np.array([[0,5,1],[5,0,0],[1,0,0]], dtype=np.float32),
        .1*np.array([[0,0,1],[0,0,0],[1,0,0]], dtype=np.float32),],
    ds=[-1., -1., -1., -3., -1.])


def animate(f, training_loop):

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim((-6, 6))
    ax.set_ylim((-6, 6))
    draw_contours(ax, f, extent=(-6,6,-6,6), num_linspace=100, num_contours=80)

    lines = [ax.plot([], [], lw=5, alpha=.5)[0] for _ in range(18)]
    xdata = [[] for _ in range(18)]
    ydata = [[] for _ in range(18)]

    def frame_processor(params):
        arr = np.concatenate((
            params["theta_a"],
            params["theta_b"]
        ))
        for i, xy in enumerate(arr):
            x, y = xy
            xdata[i].append(x)
            ydata[i].append(y)
            lines[i].set_data(xdata[i], ydata[i])
            if len(xdata[i]) > 1:
                ax.arrow(
                    xdata[i][-2],
                    ydata[i][-2],
                    0.01 * (xdata[i][-1] - xdata[i][-2]),
                    0.01 * (ydata[i][-1] - ydata[i][-2]),
                    shape='full',
                    lw=0,
                    length_includes_head=True,
                    head_width=.05,
                    color="black")

    ani = animation.FuncAnimation(
        fig,
        frame_processor,
        training_loop,
        blit=True,
        interval=10, # ms
        repeat=False)
    return ani

def opt_assert(opt_cls, opt_kwargs):
    def _print_theta(theta):
        for k, v in theta.items():
            print("  {}: {}".format(k, v.ravel()))
    initial_params = {
        "theta_piesek": .43*np.ones(shape=(2,2)),
        "theta_konik ": 4.3*np.ones(shape=(3,)),
        "theta_rybka ": .7*np.ones(shape=(1,1,1))
    }
    opt = opt_cls(initial_params=initial_params.copy(), **opt_kwargs)
    print("{} {}".format(opt_cls.__name__, opt_kwargs))
    print("initial params:")
    _print_theta(initial_params)
    for i in range(3):
        gradients = {
            "theta_piesek": (i+1)*np.arange(1,5).reshape((2,2)),
            "theta_konik ": (i+4)*np.arange(1,4).reshape((3,)),
            "theta_rybka ": (i+3)*np.ones(shape=(1,1,1))
        }
        if opt_cls.__name__ == "Nesterov":
            opt.training_phase = True
            opt.step(gradients)
            print("optimizer params (training_phase == True) - after step {}:".format(i+1))
            _print_theta(opt.get_params())
            opt.training_phase = False
            print("optimizer params (training_phase == False) - after step {}:".format(i+1))
            _print_theta(opt.get_params())
        else:
            opt.step(gradients)
            print("optimizer params - after step {}:".format(i+1))
            _print_theta(opt.get_params())
