import numpy as np
import matplotlib.pyplot as plt


def surfor_plot(ob, averaged=True):
    if averaged:
        res = [ob[name].surfor().mean(axis=0) for name in ob.names]
    else:
        res = [ob[name].surfor().sum(axis=0) for name in ob.names]
    plt.plot(np.transpose(res))
    plt.legend(ob.names)
    plt.show()

def paror_plot(ob, averaged=True):
    if averaged:
        res = [ob[name].paror().mean(axis=0) for name in ob.names]
    else:
        res = [ob[name].paror().sum(axis=0) for name in ob.names]
    plt.plot(np.transpose(res))
    plt.legend(ob.names)
    plt.show()

def logdistplot(d, **kwargs):
    import seaborn as sns
    from scipy.stats import lognorm
    ax = sns.distplot(d, fit=lognorm, **kwargs)
    shape, loc, scale = lognorm.fit(d)
    ax.set_title('Fit mode: {}'.format(loc + np.exp(np.log(scale)-shape**2)))
    plt.show()

def normdistplot(d, **kwargs):
    import seaborn as sns
    from scipy.stats import norm
    ax = sns.distplot(d, fit=norm, **kwargs)
    loc, scale = norm.fit(d)
    ax.set_title('Fit mode: {}'.format(loc))
    plt.show()