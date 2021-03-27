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
        thetagrid = kwargs.get('thetagrid', np.arange(0, 360, 10))
        ax.set_thetagrids(thetagrid, labels=thetagrid)
        ax.set_rlabel_position(0)
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
        arrow = kwargs.get('arrow', 1)
        rwidth = kwargs.get('rwidth', 1)
        for cc, val in zip(np.arange(0, 360, width), num):
            theta.extend([cc - width / 2, cc - rwidth * width / 2, cc,
                          cc + rwidth * width / 2, cc + width / 2, ])
            radii.extend([0, val * arrow, val, val * arrow, 0])
        theta = np.deg2rad(theta)
    if kwargs.get('scaled', True):
        radii = np.sqrt(radii)
    ax.fill(theta, radii, **kwargs.get('fill_kwg', {}))
    ax.set_axisbelow(True)
    if kwargs.get('show', True):
        plt.show()
    return ax


def grainsize_plot(d, **kwargs):
    # weights=None, bins='auto', left=None, right=None, num=500, alpha=95, bootstrap=False, title=None
    if 'weights' in kwargs:
        avgtxt = 'WMean'
        ylbl = 'Weighted probability density'
    else:
        avgtxt = 'Mean'
        ylbl = 'Probability density'
    weights = kwargs.get('weights', np.ones_like(d))
    bins = kwargs.get('bins', 'auto')
    bootstrap = kwargs.get('bootstrap', False)
    alpha = kwargs.get('alpha', 95)
    num = kwargs.get('num', 500)
    title = kwargs.get('title', None)
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        show = False
    else:
        f, ax = plt.subplots(figsize=kwargs.get('figsize', plt.rcParams.get('figure.figsize')))
        show = True
    d = np.asarray(d)
    ld = np.log10(d)
    bins_log = np.histogram_bin_edges(ld, bins=bins)
    bins = 10 ** bins_log
    bw = bins[1:] - bins[:-1]
    bc = (bins[:-1] + bins[1:]) / 2
    lbw = bins_log[1:] - bins_log[:-1]
    # statistics
    loc, scale = weighted_avg_and_std(ld, weights)
    rms = np.sqrt(np.mean(d ** 2))
    # default left right values
    left = kwargs.get('left', 10**(loc - 3.5 * scale))
    right = kwargs.get('right', 10**(loc + 3.5 * scale))
    # PDF
    lxx = np.linspace(np.log10(left), np.log10(right), 500)
    pdf = stats.norm.pdf(lxx, loc=loc, scale=scale)

    # hist counts
    counts, _ = np.histogram(d, bins, density=True, weights=weights)

    # bootstrap CI on mean
    if bootstrap:
        bcnt, bpdf, bmu, brms = [], [], [], []
        lxx = np.linspace(np.log10(left), np.log10(right), 500)
        for i in range(num):
            ix = np.random.choice(len(d), len(d))
            cnt, _ = np.histogram(d[ix], bins=bins, weights=weights[ix], density=True)
            bcnt.append(cnt)
            bloc, bscale = weighted_avg_and_std(np.log10(d[ix]), weights[ix])
            bpdf.append(stats.norm.pdf(lxx, loc=bloc, scale=bscale))
            bmu.append(bloc)
            brms.append(np.sqrt(np.mean(d[ix]**2)))
        # confidence intervals
        mudelta = np.array(bmu) - loc
        muconf = np.power(10, loc + np.percentile(mudelta, [(100-alpha)/2, alpha + (100-alpha)/2]))
        rmsdelta = np.array(brms) - rms
        rmsconf = rms + np.percentile(rmsdelta, [(100-alpha)/2, alpha + (100-alpha)/2])
        # plot
        ax.fill_between(10**lxx, np.min(bpdf, axis=0), np.max(bpdf, axis=0), color='lightsteelblue', alpha=0.5)
        ax.bar(bc, np.mean(bcnt, axis=0)*bw/lbw, width=0.9*bw, yerr=np.std(bcnt, axis=0)*bw/lbw/2, color='mediumseagreen')
        ax.text(0.02, 0.9, '{} EAD: {:.2f}\n{:.1f}% CI: {:.2f}-{:.2f}\nRMS EAD: {:.2f}\n{:.1f}% CI: {:.2f}-{:.2f}'.format(avgtxt, 10**loc, alpha, muconf[0], muconf[1], rms, alpha, rmsconf[0], rmsconf[1]),
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    else:
        ax.bar(bc, counts*bw/lbw, width=0.9*bw, color='mediumseagreen')
        ax.text(0.02, 0.9, '{} EAD: {:.2f}\nRMS EAD: {:.2f}'.format(avgtxt, 10**loc, rms),
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.plot(10**lxx, pdf, 'k')
    ax.set_xscale('log')
    ax.set_xlim(left=left, right=right)
    if show:
        if title is not None and show:
            f.suptitle(title)
        ax.set_ylabel(ylbl)
        plt.show()


def plot_kde(g, **kwargs):
    bins = kwargs.get('bins', 'auto')
    grouped = kwargs.get('grouped', True)
    title = kwargs.get('title', None)
    weighted = kwargs.get('weighted', True)
    bootstrap = kwargs.get('bootstrap', False)
    num = kwargs.get('num', 500)
    alpha = kwargs.get('alpha', 95)
    if weighted:
        from .utils import gaussian_kde
    else:
        from scipy.stats import gaussian_kde
    if 'ax' in kwargs:
        ax = kwargs.pop('ax')
        show = False
    else:
        f, ax = plt.subplots(figsize=kwargs.get('figsize', plt.rcParams.get('figure.figsize')))
        show = True
    ead = g.ead
    logead = np.log10(ead)
    weights = g.area
    ed = np.histogram_bin_edges(logead, bins=bins)
    xmin = 2*ed[0] - ed[1]
    xmax = 2*ed[-1] - ed[-2]
    x = np.logspace(xmin, xmax, 250)
    cntrs = 10**((ed[:-1] + ed[1:]) / 2)
    # ax.hist(logead, ed, histtype='bar', alpha=.2, normed=True, color='k', weights=weights, rwidth=0.8)
    if weighted:
        n, _ = np.histogram(logead, ed, weights=weights, density=True)
        pdf = gaussian_kde(logead, weights=weights)
    else:
        n, _ = np.histogram(logead, ed, density=True)
        pdf = gaussian_kde(logead)
    ax.bar(cntrs, n, width=0.8*np.diff(10**ed), alpha=.2, color='k')
    y = pdf(np.log10(x))
    if grouped:
        ysum = np.zeros_like(x)
        poc = 0
        for key, gg in g.class_iter():
            if weighted:
                pdf = gaussian_kde(np.log10(gg.ead), weights=gg.area)
            else:
                pdf = gaussian_kde(np.log10(gg.ead))
            yy = pdf(np.log10(x))
            ss = sum(gg.area)/sum(g.area)
            ax.fill_between(x, 0, yy*ss, label='{}'.format(key), alpha=0.4, color=g.classes.color(key))
            ysum += yy*ss
            poc += 1
        if poc > 1:
            ax.plot(x, y, label='Summed', color='k', lw=1, ls='--')
    else:
        ax.fill_between(x, 0, y, alpha=0.4, color='b')
    if weighted:
        ax.plot(x, y, label='AW-KDE', color='k', lw=2)
    else:
        ax.plot(x, y, label='KDE', color='k', lw=1, ls='--')
    ax.set_xscale('log')
    ax.legend(loc=1)
    loc, scale = weighted_avg_and_std(logead, weights)
    rms = np.sqrt(np.mean(ead**2))
    if bootstrap:
        bcnt, bmu, brms = [], [], []
        for i in range(num):
            ix = np.random.choice(len(ead), len(ead))
            cnt, _ = np.histogram(ead[ix], bins=10**ed, weights=weights[ix], density=True)
            bcnt.append(cnt)
            bloc, bscale = weighted_avg_and_std(np.log10(ead[ix]), weights[ix])
            bmu.append(bloc)
            brms.append(np.sqrt(np.mean(ead[ix]**2)))
        # confidence intervals
        mudelta = np.array(bmu) - loc
        muconf = np.power(10, loc + np.percentile(mudelta, [(100-alpha)/2, alpha + (100-alpha)/2]))
        rmsdelta = np.array(brms) - rms
        rmsconf = rms + np.percentile(rmsdelta, [(100-alpha)/2, alpha + (100-alpha)/2])
        ax.text(0.02, 0.85, 'AW mean EAD: {:.2f}\n{:.1f}% CI: {:.2f}-{:.2f}\nRMS EAD: {:.2f}\n{:.1f}% CI: {:.2f}-{:.2f}'.format(10**loc, alpha, muconf[0], muconf[1], rms, alpha, rmsconf[0], rmsconf[1]),
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    else:
        ax.text(0.02, 0.85, 'AW mean EAD: {:.2f}\nRMS EAD: {:.2f}'.format(10**loc, rms),
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    if show:
        if title is not None and show:
            f.suptitle(title)
        plt.show()
