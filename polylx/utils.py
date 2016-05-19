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
import numpy as np


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
        """Mean resultant vector

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
    def __init__(self, vals, rule='natural', k=5):
        self.rule = rule
        self.k = k
        if rule == 'equal' or rule == 'user':
            counts, bins = np.histogram(vals, k)
            index = np.digitize(vals, bins) - 1
            # if upper limit is maximum value, digitize it to last bin
            edge = len(bins) - 1
            index[np.flatnonzero(index == edge)] = edge - 1
            self.index = ['%g-%g' % (bins[i], bins[i + 1]) for i in range(len(counts))]
            self.names = np.array([self.index[i] for i in index])
        elif rule == 'natural':
            index, bins = natural_breaks(vals, k=k)
            counts = np.bincount(index)
            self.index = ['%g-%g' % (bins[i], bins[i + 1]) for i in range(len(counts))]
            self.names = np.array([self.index[i] for i in index])
        elif rule == 'jenks':
            bins = fisher_jenks(vals, k=k)
            index = np.digitize(vals, bins) - 1
            index[np.flatnonzero(index == k)] = k - 1
            counts = np.bincount(index)
            self.index = ['%g-%g' % (bins[i], bins[i + 1]) for i in range(len(counts))]
            self.names = np.array([self.index[i] for i in index])
        else:  # unique
            self.index = np.unique(vals)
            self.names = np.asarray(vals)

    def __call__(self, num):
        return np.flatnonzero(self.names == self.index[num])

    @property
    def labels(self):
        index, inverse = np.unique(self.names, return_inverse=True)
        return ['%s (%d)' % p for p in zip(index, np.bincount(inverse))]


def PolygonPath(polygon):
    """Constructs a compound matplotlib path from a Shapely object
       modified from descartes https://pypi.python.org/pypi/descartes
    """
    from matplotlib.path import Path

    def coding(ob):
        vals = np.ones(len(ob.coords), dtype=Path.code_type) * Path.LINETO
        vals[0] = Path.MOVETO
        return vals

    vertices = np.concatenate([np.asarray(polygon.exterior)] +
                              [np.asarray(r) for r in polygon.interiors])
    codes = np.concatenate([coding(polygon.exterior)] +
                           [coding(r) for r in polygon.interiors])
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


def natural_breaks(values, k=5, itmax=100):
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


def _fisher_jenks_means(values, classes=5, sort=True):
    """
    Jenks Optimal (Natural Breaks) algorithm implemented in Python.
    The original Python code comes from here:
    http://danieljlewis.org/2010/06/07/jenks-natural-breaks-algorithm-in-python/
    and is based on a JAVA and Fortran code available here:
    https://stat.ethz.ch/pipermail/r-sig-geo/2006-March/000811.html

    Returns class breaks such that classes are internally homogeneous while
    assuring heterogeneity among classes.
    Sergio J. Rey Copyright (c) 2009-10 Sergio J. Rey

    """
    if sort:
        values.sort()
    mat1 = []
    for i in range(0, len(values) + 1):
        temp = []
        for j in range(0, classes + 1):
            temp.append(0)
        mat1.append(temp)
    mat2 = []
    for i in range(0, len(values) + 1):
        temp = []
        for j in range(0, classes + 1):
            temp.append(0)
        mat2.append(temp)
    for i in range(1, classes + 1):
        mat1[1][i] = 1
        mat2[1][i] = 0
        for j in range(2, len(values) + 1):
            mat2[j][i] = float('inf')
    v = 0.0
    for l in range(2, len(values) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = float(values[i3 - 1])
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, classes + 1):
                    if mat2[l][j] >= (v + mat2[i4][j - 1]):
                        mat1[l][j] = i3
                        mat2[l][j] = v + mat2[i4][j - 1]
        mat1[l][1] = 1
        mat2[l][1] = v
    k = len(values)
    kclass = []
    for i in range(0, classes + 1):
        kclass.append(0)
    kclass[classes] = float(values[len(values) - 1])
    kclass[0] = float(values[0])
    countNum = classes
    while countNum >= 2:
        pivot = mat1[k][countNum]
        id = int(pivot - 2)
        kclass[countNum - 1] = values[id]
        k = int(pivot - 1)
        countNum -= 1
    return kclass


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


def densify(x, y, repeat=1):
    for i in range(repeat):
        x = np.insert(x, np.s_[1:], x[:-1] + np.diff(x) / 2)
        y = np.insert(y, np.s_[1:], y[:-1] + np.diff(y) / 2)
    return x, y


def _chaikin_ring(x, y, repeat=4):
    # Chaikin's corner cutting algorithm
    for i in range(repeat):
        x, y = densify(x, y, 2)
        x = np.append(x[1::2], x[1])
        y = np.append(y[1::2], y[1])
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


def _visvalingam_whyatt_ring(x, y, minarea=None):
    do = True
    tot = 0
    while do:
        xx = np.concatenate((x[-2:-1], x, x[1:2]))
        yy = np.concatenate((y[-2:-1], y, y[1:2]))
        i0 = np.arange(len(xx) - 2)
        i1 = i0 + 1
        i2 = i0 + 2
        a = (xx[i0] * (yy[i1] - yy[i2]) + xx[i1] * (yy[i2] - yy[i0]) + xx[i2] * (yy[i0] - yy[i1])) / 2
        ix = abs(a).argmin()
        mn = a[ix]
        if abs(tot + mn) < minarea and len(x) > 4:
            if ix == 0 or ix == len(x) - 1:
                x = np.concatenate((x[1:-1], x[1:2]))
                y = np.concatenate((y[1:-1], y[1:2]))
            else:
                x = np.delete(x, ix)
                y = np.delete(y, ix)
            tot += mn
        else:
            do = False
    return x, y
