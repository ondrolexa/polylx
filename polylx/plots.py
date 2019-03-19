import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from .core import PolySet
from .utils import weighted_avg_and_std

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
    kwargs['fit'] = stats.lognorm
    if 'kde' not in kwargs:
        kwargs['kde'] = False
    ax = sns.distplot(d, **kwargs)
    shape, loc, scale = stats.lognorm.fit(d)
    mode = loc + np.exp(np.log(scale) - shape**2)
    sts = np.asarray(stats.lognorm.stats(shape, loc=loc, scale=scale, moments='mv'))
    ax.set_title('Mode:{:g} Mean:{:g} Var:{:g}'.format(mode, *sts))
    plt.show()


def normdist_plot(d, **kwargs):
    kwargs['fit'] = stats.norm
    if 'kde' not in kwargs:
        kwargs['kde'] = False
    ax = sns.distplot(d, **kwargs)
    loc, scale = stats.norm.fit(d)
    sts = np.asarray(stats.norm.stats(loc=loc, scale=scale, moments='mv'))
    ax.set_title('Mean:{:g} Var:{:g}'.format(*sts))
    plt.show()


def rose_plot(ang, **kwargs):
    if 'ax' in kwargs:
            ax = kwargs.pop('ax')
    else:
        fig = plt.figure(figsize=kwargs.get('figsize', plt.rcParams.get('figure.figsize')))
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
    if kwargs.get('show', True):
        plt.show()
    return ax

def grainsize_plot(d, weights=None, bins='auto', left=None, right=None, num=500, alpha=95, bootstrap=False, title=None):
    d = np.asarray(d)
    if weights is None:
        weights = np.ones_like(d)
    ld = np.log10(d)
    bins_log = np.histogram_bin_edges(ld, bins=bins)
    bins = 10**bins_log
    bw = bins[1:] - bins[:-1]
    bc = (bins[:-1] + bins[1:])/2
    lbw = bins_log[1:] - bins_log[:-1]
    # statistics
    loc, scale = weighted_avg_and_std(ld, weights)

    # default left right values
    if left is None:
        left = 10**(loc - 3.5*scale)
    if right is None:
        right = 10**(loc + 3.5*scale)
    # PDF
    lxx = np.linspace(np.log10(left), np.log10(right), 500)
    pdf = stats.norm.pdf(lxx, loc=loc, scale=scale)

    # hist counts
    counts,_ = np.histogram(d, bins, density=True, weights=weights)

    # plot
    f, ax = plt.subplots(figsize=(9, 5))
    if title is not None:
        f.suptitle(title)

    # bootstrap CI on mean
    if bootstrap:
        bcnt, bpdf, mu = [], [], []
        for i in range(num):
            ix = np.random.choice(len(d), len(d))
            cnt, _ = np.histogram(d[ix], bins=bins, weights=weights[ix], density=True)
            bcnt.append(cnt)
            loc, scale = weighted_avg_and_std(np.log10(d[ix]), weights[ix])
            bpdf.append(stats.norm.pdf(lxx, loc=loc, scale=scale))
            mu.append(loc)
        # confidence interval
        delta = np.array(mu) - loc
        conf = np.power(10, loc + np.percentile(delta, [(100-alpha)/2, alpha + (100-alpha)/2]))
        # plot
        ax.fill_between(10**lxx, np.min(bpdf, axis=0), np.max(bpdf, axis=0), color='lightsteelblue', alpha=0.5)
        ax.bar(bc, np.mean(bcnt, axis=0)*bw/lbw, width=0.9*bw, yerr=np.std(bcnt, axis=0)*bw/lbw/2, color='mediumseagreen')
        ax.text(0.02, 0.9, 'AW mean EAD: {:.2f}\n{:.1f}% CI: {:.2f}-{:.2f}'.format(10**loc, alpha, *conf),
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    else:
        ax.bar(bc, counts*bw/lbw, width=0.9*bw, color='mediumseagreen')
        ax.text(0.02, 0.9, 'AW mean EAD: {:.2f}'.format(10**loc),
             horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.plot(10**lxx, pdf, 'k')
    ax.set_xscale('log')
    ax.set_xlim(left=left, right=right)
    plt.show()