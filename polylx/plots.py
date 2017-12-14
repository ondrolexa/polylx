import numpy as np
import matplotlib.pyplot as plt
from .core import PolySet

##########################
# Plots for polylx objects
##########################


def surfor_plot(ob, averaged=True):
    assert isinstance(ob, PolySet), ('First argument must be Grains or Boundaries instance.')
    for key, g in ob.class_iter():
        if averaged:
            res = g.surfor().mean(axis=0)
        else:
            res = g.surfor().sum(axis=0)
        plt.plot(res, color=ob.classes.color(key), label=key)
    plt.legend()
    plt.show()


def paror_plot(ob, averaged=True):
    assert isinstance(ob, PolySet), ('First argument must be Grains or Boundaries instance.')
    for key, g in ob.class_iter():
        if averaged:
            res = g.paror().mean(axis=0)
        else:
            res = g.paror().sum(axis=0)
        plt.plot(res, color=ob.classes.color(key), label=key)
    plt.legend()
    plt.show()

#############
# Other plots
#############


def logdist_plot(d, **kwargs):
    import seaborn as sns
    from scipy.stats import lognorm
    ax = sns.distplot(d, fit=lognorm, **kwargs)
    shape, loc, scale = lognorm.fit(d)
    mode = loc + np.exp(np.log(scale) - shape**2)
    stats = np.asarray(lognorm.stats(shape, loc=loc, scale=scale, moments='mv'))
    ax.set_title('Mode:{:g} Mean:{:g} Var:{:g}'.format(mode, *stats))
    plt.show()


def normdist_plot(d, **kwargs):
    import seaborn as sns
    from scipy.stats import norm
    ax = sns.distplot(d, fit=norm, **kwargs)
    loc, scale = norm.fit(d)
    stats = np.asarray(norm.stats(loc=loc, scale=scale, moments='mv'))
    ax.set_title('Mean:{:g} Var:{:g}'.format(*stats))
    plt.show()


def rose_plot(ang, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    if kwargs.get('pdf', False):
        from scipy.stats import vonmises
        theta = np.linspace(-np.pi, np.pi, 1801)
        radii = np.zeros_like(theta)
        kappa = kwargs.get('kappa', 250)
        for a in ang:
            radii += vonmises.pdf(theta, kappa, loc=np.radians(a))
            radii += vonmises.pdf(theta, kappa, loc=np.radians(a + 180))
        radii /= len(ang)
    else:
        bins = kwargs.get('bins', 36)
        width = 360 / bins
        if 'weights' in kwargs:
            num, bin_edges = np.histogram(np.concatenate((ang, ang + 180)),
                                          bins=bins + 1,
                                          range=(-width / 2, 360 + width / 2),
                                          weights=np.concatenate((kwargs.get('weights'), kwargs.get('weights'))),
                                          density=kwargs.get('density', False))
        else:
            num, bin_edges = np.histogram(np.concatenate((ang, ang + 180)),
                                          bins=bins + 1,
                                          range=(-width / 2, 360 + width / 2),
                                          density=kwargs.get('density', False))
        num[0] += num[-1]
        num = num[:-1]
        theta, radii = [], []
        arrow = kwargs.get('arrow', 0.95)
        rwidth = kwargs.get('rwidth', 1)
        for cc, val in zip(np.arange(0, 360, width), num):
            theta.extend([cc - width / 2, cc - rwidth * width / 2, cc,
                          cc + rwidth * width / 2, cc + width / 2, ])
            radii.extend([0, val * arrow, val, val * arrow, 0])
        theta = np.deg2rad(theta)
    if kwargs.get('scaled', False):
        radii = np.sqrt(radii)
    ax.fill(theta, radii, **kwargs.get('fill_kwg', {}))
    plt.show()
    return ax
