# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:42:54 2014

@author: Ondrej Lexa

Example:

from polylx.utils import optimize_colormap
g.plot(cmap=optimize_colormap('jet'))

# use circular statistics for agg
g.groups('lao').agg(circular.csd)
"""
from __future__ import division
from copy import deepcopy
import numpy as np
import matplotlib.cm as cm
import seaborn as sns


def fixzero(x):
    return x * (x > np.finfo(float).eps)


def fixratio(x, y):
    if y == 0:
        return np.inf
    else:
        return x / y


class circular(object):
    @staticmethod
    def rho(x):
        """Mean resultant vector as complex number

        """
        return np.exp(2j * np.deg2rad(x)).mean()

    @staticmethod
    def R(x):
        """Length of mean resultant vector

        """
        return abs(circular.rho(x))

    @staticmethod
    def mean(x):
        """Mean direction

        """
        return np.angle(circular.rho(x), deg=True) / 2 % 180

    @staticmethod
    def var(x):
        """Circular variance

        """
        return 1 - circular.R(x)

    @staticmethod
    def csd(x):
        """Circular standard deviation

        """
        return np.sqrt(-2 * np.log(circular.R(x)))

    @staticmethod
    def angdev(x):
        """Angular deviation

        """
        return np.sqrt(2 * circular.var(x))

    @staticmethod
    def mean_conf(x, cl=0.95):
        """Confidence limit on mean

        cl confidence on mean between mu-conf..mu+conf

        """
        from scipy.stats import gamma
        n = len(x)
        R = circular.R(x)
        r = R * n
        c2 = gamma.ppf(cl, 0.5, scale=2)
        if R < .9 and R > np.sqrt(c2 / 2 / n):
            t = np.sqrt((2 * n * (2 * r**2 - n * c2)) / (4 * n - c2))
        elif R >= .9:
            t = np.sqrt(n**2 - (n**2 - r**2) * np.exp(c2 / n))
        else:  # Resultant vector does not allow to specify confidence limits
            t = np.nan
        return deg.acos(t / r)

    @staticmethod
    def angskew(x):
        """Angular skewness

        """
        return np.mean(deg.sin(2 * circular.circdist(x, circular.mean(x))))

    @staticmethod
    def sas(x):
        """Standardized angular skewness

        """
        rho_p, mu_p = circular.circmoment(x, 2)
        s = deg.sin(circular.circdist(mu_p, 2 * circular.mean(x)))
        d = (1 - circular.var(x))**(2 / 3)
        return rho_p * s / d

    @staticmethod
    def circdist(x, y):
        """Pairwise difference around the circle

        """
        return np.angle(np.exp(1j * x) / np.exp(1j * y), deg=True)

    @staticmethod
    def circmoment(x, p=1):
        """Complex centered p-th moment

        """
        mp = np.exp(2j * p * np.deg2rad(x)).mean()
        return np.abs(mp), np.angle(mp, deg=True)


class deg(object):
    @staticmethod
    def sin(x):
        return np.sin(np.deg2rad(x))

    @staticmethod
    def cos(x):
        return np.cos(np.deg2rad(x))

    @staticmethod
    def tan(x):
        return np.tan(np.deg2rad(x))

    @staticmethod
    def asin(x):
        return np.rad2deg(np.arcsin(x))

    @staticmethod
    def acos(x):
        return np.rad2deg(np.arccos(x))

    @staticmethod
    def atan(x):
        return np.rad2deg(np.arctan(x))

    @staticmethod
    def atan2(x1, x2):
        return np.rad2deg(np.arctan2(x1, x2))


class Classify(object):
    """Class to store classification and colortable for legend

    """
    def __init__(self, vals, **kwargs):
        rule = kwargs.get('rule', 'natural')
        k = kwargs.get('k', 5)
        label = kwargs.get('label', 'Default')
        cmap = kwargs.get('cmap', 'viridis')
        self.vals = vals
        self.rule = rule
        self.label = label
        if rule == 'equal' or rule == 'user':
            counts, bins = np.histogram(vals, k)
            index = np.digitize(vals, bins) - 1
            # if upper limit is maximum value, digitize it to last bin
            edge = len(bins) - 1
            index[np.flatnonzero(index == edge)] = edge - 1
            prec = int(max(-np.floor(np.log10(np.diff(bins).min())) + 1, 0))
            self.index = ['{:.{prec}f}-{:.{prec}f}'.format(bins[i], bins[i + 1], prec=prec) for i in range(len(counts))]
            self.names = np.array([self.index[i] for i in index])
        elif rule == 'natural':
            index, bins = natural_breaks(vals, k=k)
            counts = np.bincount(index)
            prec = int(max(-np.floor(np.log10(np.diff(bins).min())) + 1, 0))
            self.index = ['{:.{prec}f}-{:.{prec}f}'.format(bins[i], bins[i + 1], prec=prec) for i in range(len(counts))]
            self.names = np.array([self.index[i] for i in index])
        elif rule == 'jenks':
            bins = fisher_jenks(vals, k=k)
            index = np.digitize(vals, bins) - 1
            index[np.flatnonzero(index == k)] = k - 1
            counts = np.bincount(index)
            prec = int(max(-np.floor(np.log10(np.diff(bins).min())) + 1, 0))
            self.index = ['{:.{prec}f}-{:.{prec}f}'.format(bins[i], bins[i + 1], prec=prec) for i in range(len(counts))]
            self.names = np.array([self.index[i] for i in index])
        else:  # unique
            self.index = list(np.unique(vals))
            self.names = np.asarray(vals)
            # other cmap for unique
            cmap = kwargs.get('cmap', self.sns2cmap('muted'))
        self.set_cmap(cmap)

    def __call__(self, num):
        return np.flatnonzero(self.names == self.index[num])

    def __getitem__(self, index):
        cl = deepcopy(self)
        cl.vals = [self.vals[ix] for ix in index]
        cl.names = self.names[index]
        cl.index = list(np.unique(cl.names))
        return cl

    def __repr__(self):
        tmpl = 'Classification %s of %s with %g classes.'
        return tmpl % (self.rule, self.label, len(self.index))

    @property
    def labels(self):
        index, inverse = np.unique(self.names, return_inverse=True)
        return ['%s (%d)' % p for p in zip(index, np.bincount(inverse))]

    def set_cmap(self, cmap):
        """Set colormap for actual classification.

        Args:
          cmap: matplotlib ListedColormap

        """
        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
        n = len(self.index)
        if n > 1:
            pos = np.round(np.linspace(0, cmap.N - 1, n))
        else:
            pos = [127]
        self._colors_dict = dict(zip(self.index, [cmap(int(i)) for i in pos]))

    def sns2cmap(self, palette):
        """Create matplotlib ListedColormap from seaborn pallete

        Args:
          palette: name of seaborn palette or list of colors

        """
        from matplotlib.colors import ListedColormap
        if isinstance(palette, str):
            palette = sns.color_palette(palette, len(self.index))
        return ListedColormap(sns.color_palette(palette))

    def set_colortable(self, ct):
        """Update colors for actual classification from dictionary.

        Args:
          ct: dictionary with color definition

        """
        if isinstance(ct, dict):
            self._colors_dict.update(ct)

    @property
    def colortable(self):
        """Get dictionary of colors used in actual classification.

        """
        return deepcopy(self._colors_dict)

    def color(self, key):
        return self._colors_dict.get(key, (0, 0, 0))


def PolygonPath(polygon):
    """Constructs a compound matplotlib path from a Shapely object
       modified from descartes https://pypi.python.org/pypi/descartes
    """
    from matplotlib.path import Path

    def coding(ob):
        vals = np.ones(len(ob.coords), dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    vertices = np.concatenate([np.asarray(polygon.exterior)] + [np.asarray(r) for r in polygon.interiors])
    codes = np.concatenate([coding(polygon.exterior)] + [coding(r) for r in polygon.interiors])
    return Path(vertices, codes)


def optimize_colormap(name):
    # optimize lightness to the desired value
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap
    from colormath.color_objects import LabColor, sRGBColor
    from colormath.color_conversions import convert_color
    cmap = cm.get_cmap(name)
    values = cmap(np.linspace(0, 1, 256))[:, :3]
    lab_colors = []
    for rgb in values:
        lab_colors.append(convert_color(sRGBColor(*rgb), target_cs=LabColor))

    target_lightness = np.ones(256) * np.mean([_i.lab_l for _i in lab_colors])
    for color, lightness in zip(lab_colors, target_lightness):
        color.lab_l = lightness

    # Go back to rbg.
    rgb_colors = [convert_color(_i, target_cs=sRGBColor) for _i in lab_colors]
    # Clamp values as colorspace of LAB is larger then sRGB.
    rgb_colors = [(_i.clamped_rgb_r,
                   _i.clamped_rgb_g,
                   _i.clamped_rgb_b) for _i in rgb_colors]
    cmap = LinearSegmentedColormap.from_list(name=name + "_optimized",
                                             colors=rgb_colors)
    return cmap


def natural_breaks(values, k=5, itmax=1000):
    """
    natural breaks helper function
    Sergio J. Rey Copyright (c) 2009-10 Sergio J. Rey
    """
    values = np.array(values)
    uv = np.unique(values)
    uvk = len(uv)
    if uvk < k:
        print('Warning: Not enough unique values in array to form k classes')
        print('Warning: setting k to %d' % uvk)
        k = uvk
    sids = np.random.permutation(range(len(uv)))[0:k]
    seeds = uv[sids]
    seeds.sort()
    diffs = abs(np.matrix([values - seed for seed in seeds]))
    c0 = diffs.argmin(axis=0)
    c0 = np.array(c0)[0]
    solving = True
#    solved = False
    rk = range(k)
    it = 0
    while solving:
        # get centroids of clusters
        seeds = [np.median(values[c0 == c]) for c in rk]
        seeds.sort()
        # for each value find closest centroid
        diffs = abs(np.matrix([values - seed for seed in seeds]))
        # assign value to that centroid
        c1 = diffs.argmin(axis=0)
        c1 = np.array(c1)[0]
        # compare new classids to previous
        d = abs(c1 - c0)
        if d.sum() == 0:
            solving = False
#            solved = True
        else:
            c0 = c1
        it += 1
        if it == itmax:
            solving = False
    cuts = [min(values)] + [max(values[c1 == c]) for c in rk]
    return c1, cuts


def fisher_jenks(values, k=5):
    """
    Our own version of Jenks Optimal (Natural Breaks) algorithm
    implemented in Python. The implementation follows the original
    procedure described in the book, which is a two-phased approach.
    First phase aims at calculating the variance matrix between the
    ith and jth element in the data array;
    Second phase runs iteratively to construct the optimal K-partition
    from results of K-1 - partitions.
    Sergio J. Rey Copyright (c) 2009-10 Sergio J. Rey
    """

    values = np.sort(values)
    numVal = len(values)

    varMat = (numVal + 1) * [0]
    for i in range(numVal + 1):
        varMat[i] = (numVal + 1) * [0]

    errorMat = (numVal + 1) * [0]
    for i in range(numVal + 1):
        errorMat[i] = (k + 1) * [float('inf')]

    pivotMat = (numVal + 1) * [0]
    for i in range(numVal + 1):
        pivotMat[i] = (k + 1) * [0]

    # building up the initial variance matrix
    for i in range(1, numVal + 1):
        sumVals = 0
        sqVals = 0
        numVals = 0
        for j in range(i, numVal + 1):
            val = float(values[j - 1])
            sumVals += val
            sqVals += val * val
            numVals += 1.0
            varMat[i][j] = sqVals - sumVals * sumVals / numVals
            if i == 1:
                errorMat[j][1] = varMat[i][j]

    for cIdx in range(2, k + 1):
        for vl in range(cIdx - 1, numVal):
            preError = errorMat[vl][cIdx - 1]
            for vIdx in range(vl + 1, numVal + 1):
                curError = preError + varMat[vl + 1][vIdx]
                if errorMat[vIdx][cIdx] > curError:
                    errorMat[vIdx][cIdx] = curError
                    pivotMat[vIdx][cIdx] = vl

    pivots = (k + 1) * [0]
    pivots[k] = values[numVal - 1]
    pivots[0] = values[0]
    lastPivot = pivotMat[numVal][k]

    pNum = k - 1
    while pNum > 0:
        pivots[pNum] = values[lastPivot - 1]
        lastPivot = pivotMat[lastPivot][pNum]
        pNum -= 1

    return pivots


def find_ellipse(x, y):
    # direct ellipse fit
    xmean = x.mean()
    ymean = y.mean()
    x -= xmean
    y -= ymean
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    q = V[:, n]
    # get parameters
    b, c, d, f, g, a = q[1] / 2, q[2], q[3] / 2, q[4] / 2, q[5], q[0]
    num = b * b - a * c
    xc = (c * d - b * f) / num + xmean
    yc = (a * f - b * d) / num + ymean
    phi = 0.5 * np.arctan(2 * b / (a - c))
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    de = np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a)
    down1 = (b * b - a * c) * ((c - a) * de)
    down2 = (b * b - a * c) * ((a - c) * de)
    a = np.sqrt(up / down1)
    b = np.sqrt(up / down2)
    return xc, yc, phi, a, b


def inertia_moments(x, y, xc, yc):
    x = x - xc
    y = y - yc
    d = x[1:] * y[:-1] - x[:-1] * y[1:]
    A = np.sum(d) / 2
    ax = np.sum(d * (x[1:] + x[:-1])) / (6 * A)
    ay = np.sum(d * (y[1:] + y[:-1])) / (6 * A)
    axx = np.sum(d * (x[1:]**2 + x[1:] * x[:-1] + x[:-1]**2)) / (12 * A)
    ayy = np.sum(d * (y[1:]**2 + y[1:] * y[:-1] + y[:-1]**2)) / (12 * A)
    axy = np.sum(d * (2 * x[1:] * y[1:] + x[1:] * y[:-1] + x[:-1] * y[1:] + 2 * x[:-1] * y[:-1])) / (24 * A)
    mxx = axx - ax**2
    myy = ayy - ay**2
    mxy = axy - ax * ay
    return np.array([mxx, myy, mxy])


def densify(x, y, repeat=1):
    for i in range(repeat):
        x = np.insert(x, np.s_[1:], x[:-1] + np.diff(x) / 2)
        y = np.insert(y, np.s_[1:], y[:-1] + np.diff(y) / 2)
    return x, y


def _chaikin(x, y, repeat=2, is_ring=False):
    # Chaikin's corner cutting algorithm
    for i in range(repeat):
        x, y = densify(x, y, 2)
        if is_ring:
            x = np.append(x[1::2], x[1])
            y = np.append(y[1::2], y[1])
        else:
            x = np.concatenate((x[:1], x[3:-3:2], x[-1:]))
            y = np.concatenate((y[:1], y[3:-3:2], y[-1:]))
    return x, y


def _spline_ring(x, y, densify=5, pad=5):
    from scipy.interpolate import spline
    num = len(x)
    # padding
    x = np.concatenate((x[-pad:-1], x, x[1:pad]))
    y = np.concatenate((y[-pad:-1], y, y[1:pad]))
    # distance parameter normalized on beginning and end
    t = np.zeros(x.shape)
    t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
    t = np.cumsum(t)
    t -= t[pad - 1]
    t /= t[-pad]
    nt = np.linspace(0, 1, num * densify)
    x = spline(t, x, nt)
    y = spline(t, y, nt)
    x[-1] = x[0]
    y[-1] = y[0]
    return x, y


def _visvalingam_whyatt(x, y, threshold=1, is_ring=False):
    ds = np.c_[x[1:] - x[:-1], y[1:] - y[:-1]]
    l0 = np.sum(np.linalg.norm(ds, axis=1))
    if is_ring:
        do = len(x) > 4
        while do:
            xx = np.concatenate((x[-2:-1], x, x[1:2]))
            yy = np.concatenate((y[-2:-1], y, y[1:2]))
            i0 = np.arange(len(xx) - 2)
            i1 = i0 + 1
            i2 = i0 + 2
            a = (xx[i0] * (yy[i1] - yy[i2]) + xx[i1] * (yy[i2] - yy[i0]) + xx[i2] * (yy[i0] - yy[i1])) / 2
            ix = abs(a).argmin()
            if ix == 0 or ix == len(x) - 1:
                xx = np.concatenate((x[1:-1], x[1:2]))
                yy = np.concatenate((y[1:-1], y[1:2]))
            else:
                xx = np.delete(x, ix)
                yy = np.delete(y, ix)
            ds = np.c_[xx[1:] - xx[:-1], yy[1:] - yy[:-1]]
            l1 = np.sum(np.linalg.norm(ds, axis=1))
            dif = 100 * abs(l1 - l0) / l0
            if dif < threshold and len(xx) > 4:
                x = xx
                y = yy
            else:
                do = False
    else:
        do = len(x) > 2
        while do:
            i0 = np.arange(len(x) - 2)
            i1 = i0 + 1
            i2 = i0 + 2
            a = (x[i0] * (y[i1] - y[i2]) + x[i1] * (y[i2] - y[i0]) + x[i2] * (y[i0] - y[i1])) / 2
            ix = abs(a).argmin()
            xx = np.delete(x, ix + 1)
            yy = np.delete(y, ix + 1)
            ds = np.c_[xx[1:] - xx[:-1], yy[1:] - yy[:-1]]
            l1 = np.sum(np.linalg.norm(ds, axis=1))
            dif = 100 * abs(l1 - l0) / l0
            if dif < threshold and len(xx) > 2:
                x = xx
                y = yy
            else:
                do = False
    return x, y


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))


def classify_shapes(g, **kwargs):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    N = kwargs.get('N', 128)
    X = StandardScaler().fit_transform(g.shape_vector(N=N))
    pca = PCA(n_components=N // 2)
    pcs = pca.fit_transform(X)
    # Y = StandardScaler().fit_transform(np.array([pcs.T[0], np.log10(g.ead)]).T)
    Y = np.append(pcs[:, :kwargs.get('n_pcas', 1)], np.atleast_2d(np.log10(g.ead)).T, axis=1)
    Z = StandardScaler().fit_transform(Y)
    kmeans = KMeans(n_clusters=kwargs.get('n_clusters', 2),
                    init=kwargs.get('init', 'k-means++'),
                    max_iter=kwargs.get('max_iter', 300),
                    n_init=kwargs.get('n_init', 10),
                    random_state=kwargs.get('random_state', 0))
    classes = kmeans.fit_predict(Z)
    g.classify(classes, rule='unique')


class gaussian_kde(object):
    """Representation of a kernel-density estimate using Gaussian kernels.

    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `gaussian_kde` works for both uni-variate and multi-variate data.   It
    includes automatic bandwidth determination.  The estimation works best for
    a unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of dims, # of data).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `gaussian_kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, shape (n, ), optional, default: None
        An array of weights, of the same shape as `x`.  Each value in `x`
        only contributes its associated weight towards the bin count
        (instead of 1).

    Attributes
    ----------
    dataset : ndarray
        The dataset with which `gaussian_kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : float
        Effective sample size using Kish's approximation.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.

    Methods
    -------
    kde.evaluate(points) : ndarray
        Evaluate the estimated pdf on a provided set of points.
    kde(points) : ndarray
        Same as kde.evaluate(points)
    kde.pdf(points) : ndarray
        Alias for ``kde.evaluate(points)``.
    kde.set_bandwidth(bw_method='scott') : None
        Computes the bandwidth, i.e. the coefficient that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        .. versionadded:: 0.11.0
    kde.covariance_factor : float
        Computes the coefficient (`kde.factor`) that multiplies the data
        covariance matrix to obtain the kernel covariance matrix.
        The default is `scotts_factor`.  A subclass can overwrite this method
        to provide a different method, or set it through a call to
        `kde.set_bandwidth`.

    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),

    with ``n`` the number of data points and ``d`` the number of dimensions.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::

        (n * (d + 2) / 4.)**(-1. / (d + 4)).

    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.

    Examples
    --------
    Generate some random two-dimensional data:

    >>> from scipy import stats
    >>> def measure(n):
    >>>     "Measurement model, return two coupled measurements."
    >>>     m1 = np.random.normal(size=n)
    >>>     m2 = np.random.normal(scale=0.5, size=n)
    >>>     return m1+m2, m1-m2

    >>> m1, m2 = measure(2000)
    >>> xmin = m1.min()
    >>> xmax = m1.max()
    >>> ymin = m2.min()
    >>> ymax = m2.max()

    Perform a kernel density estimate on the data:

    >>> X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    >>> positions = np.vstack([X.ravel(), Y.ravel()])
    >>> values = np.vstack([m1, m2])
    >>> kernel = stats.gaussian_kde(values)
    >>> Z = np.reshape(kernel(positions).T, X.shape)

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
    ...           extent=[xmin, xmax, ymin, ymax])
    >>> ax.plot(m1, m2, 'k.', markersize=2)
    >>> ax.set_xlim([xmin, xmax])
    >>> ax.set_ylim([ymin, ymax])
    >>> plt.show()

    """
    def __init__(self, dataset, bw_method=None, weights=None):
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")
        self.d, self.n = self.dataset.shape

        if weights is not None:
            self.weights = weights / np.sum(weights)
        else:
            self.weights = np.ones(self.n) / self.n

        # Compute the effective sample size
        # http://surveyanalysis.org/wiki/Design_Effects_and_Effective_Sample_Size#Kish.27s_approximate_formula_for_computing_effective_sample_size
        self.neff = 1.0 / np.sum(self.weights ** 2)

        self.set_bandwidth(bw_method=bw_method)

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        from scipy.spatial.distance import cdist

        points = np.atleast_2d(points)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d, self.d)
                raise ValueError(msg)

        # compute the normalised residuals
        chi2 = cdist(points.T, self.dataset.T, 'mahalanobis', VI=self.inv_cov) ** 2
        # compute the pdf
        result = np.sum(np.exp(-.5 * chi2) * self.weights, axis=1) / self._norm_factor

        return result

    __call__ = evaluate

    def scotts_factor(self):
        return np.power(self.neff, -1./(self.d+4))

    def silverman_factor(self):
        return np.power(self.neff*(self.d+2.0)/4.0, -1./(self.d+4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor

    def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.

        Notes
        -----
        .. versionadded:: 0.11

        Examples
        --------
        >>> x1 = np.array([-7, -5, 1, 4, 5.])
        >>> kde = stats.gaussian_kde(x1)
        >>> xs = np.linspace(-10, 10, num=50)
        >>> y1 = kde(xs)
        >>> kde.set_bandwidth(bw_method='silverman')
        >>> y2 = kde(xs)
        >>> kde.set_bandwidth(bw_method=kde.factor / 3.)
        >>> y3 = kde(xs)

        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> ax.plot(x1, np.ones(x1.shape) / (4. * x1.size), 'bo',
        ...         label='Data points (rescaled)')
        >>> ax.plot(xs, y1, label='Scott (default)')
        >>> ax.plot(xs, y2, label='Silverman')
        >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')
        >>> ax.legend()
        >>> plt.show()

        """
        from six import string_types

        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, string_types):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            # Compute the mean and residuals
            _mean = np.sum(self.weights * self.dataset, axis=1)
            _residual = (self.dataset - _mean[:, None])
            # Compute the biased covariance
            self._data_covariance = np.atleast_2d(np.dot(_residual * self.weights, _residual.T))
            # Correct for bias (http://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance)
            self._data_covariance /= (1 - np.sum(self.weights ** 2))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.covariance))  # * self.n
