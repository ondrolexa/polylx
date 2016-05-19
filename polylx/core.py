# -*- coding: utf-8 -*-
"""
Python module to visualize and analyze digitized 2D microstructures.

@author: Ondrej Lexa

Examples:
  >>> from polylx import *
  >>> g = Grains.from_shp('')
  >>> b = g.boundaries()

"""
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import shape, Polygon, LinearRing
import networkx as nx
import pandas as pd

from .shapefile import Reader
from .utils import fixratio, fixzero, deg, Classify, PolygonPath
from .utils import find_ellipse, densify
from .utils import _chaikin_ring, _spline_ring, _visvalingam_whyatt_ring

from pkg_resources import resource_filename

respath = resource_filename(__name__, 'example')


class PolyShape(object):
    """Base class to store polygon or polyline

    Properties:
      shape: ``shapely.geometry`` object
      name: name of polygon or polyline.
      fid: feature id

    Note that all properties from ``shapely.geometry`` object are inherited.

    """
    def __init__(self, shape, name, fid):
        self.shape = shape
        self.name = str(name)
        self.fid = int(fid)

    def __getattr__(self, attr):
        if hasattr(self.shape, attr):
            return getattr(self.shape, attr)
        else:
            raise AttributeError

    @property
    def shape_method(self):
        """Returns shape method in use

        """
        return self._shape_method

    @shape_method.setter
    def shape_method(self, value):
        """Sets shape method

        """
        getattr(self, value)()   # to evaluate and check validity

    @property
    def ar(self):
        """Returns axial ratio

        Note that axial ratio is calculated from long and short axes
        calculated by actual ``shape method``.

        """
        return fixratio(self.la, self.sa)

    @property
    def ma(self):
        """Returns mean axis

        Mean axis is calculated as square root of long axis multiplied by
        short axis. Both axes are calculated by actual ``shape method``.

        """
        return np.sqrt(self.la * self.sa)

    @property
    def centroid(self):
        """Returns the geometric center of the object

        """
        return self.shape.centroid.coords[0]

    def feret(self, angle=0):
        """Returns the ferret diameter for given angle

        Args:
          angle: angle of caliper rotation

        """
        pp = np.dot(self.hull.T, np.array([deg.sin(angle), deg.cos(angle)]))
        return pp.max(axis=0) - pp.min(axis=0)

    #################################################################
    # Common shape methods (should modify sa, la, sao, lao, xc, yc) #
    #################################################################
    def maxferet(self):
        """`shape_method`: maxferet

        Long axis is defined as the maximum caliper of the polygon/polyline.
        Short axis correspond to caliper orthogonal to long axis.
        Center coordinates are set to centroid.

        """
        xy = self.hull.T
        pa = np.array(list(itertools.combinations(range(len(xy)), 2)))
        d2 = np.sum((xy[pa[:, 0]] - xy[pa[:, 1]])**2, axis=1)
        ix = d2.argmax()
        dxy = xy[pa[ix][1]] - xy[pa[ix][0]]
        self.la = np.sqrt(np.max(d2))
        self.lao = deg.atan2(*dxy) % 180
        self.sao = (self.lao + 90) % 180
        self.sa = fixzero(self.feret(self.sao))
        self.xc, self.yc = self.shape.centroid.coords[0]
        self._shape_method = 'maxferet'


class Grain(PolyShape):
    """Grain class to store polygonal grain geometry

    A two-dimensional grain bounded by a linear ring with non-zero area.
    It may have one or more negative-space “holes” which are also bounded
    by linear rings.

    Properties:
      shape: ``shapely.geometry.polygon.Polygon`` object
      name: string with phase name. Default "None"
      fid: feature id. Default 0
      shape_method: Method to calculate axes and orientation

    """
    def __init__(self, shape, name='None', fid=0):
        """Create ``Grain`` object

        """
        super(Grain, self).__init__(shape, name, fid)
        self.shape_method = 'moment'

    def __repr__(self):
        return ('Grain {g.fid:d} [{g.name:s}] '
                'A:{g.area:g}, AR:{g.ar:g}, '
                'LAO:{g.lao:g} ({g.shape_method:s})').format(g=self)

    @classmethod
    def from_coords(self, x, y, name='None', fid=0):
        """Create ``Grain`` from coordinate arrays

        Example:
          >>> g=Grain.from_coords([0,0,2,2],[0,1,1,0])
          >>> g.xy
          array([[ 0.,  0.,  2.,  2.,  0.],
                 [ 0.,  1.,  1.,  0.,  0.]])

        """
        geom = Polygon([(xx, yy) for xx, yy in zip(x, y)])
        # try  to "clean" self-touching or self-crossing polygons
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_valid and geom.geom_type == 'Polygon':
            return self(geom, name, fid)
        else:
            print('Invalid geometry.')

    @property
    def xy(self):
        """Returns array of vertex coordinate pair.

        Note that only vertexes from exterior boundary are returned.

        """
        return np.array(self.shape.exterior.xy)

    @property
    def hull(self):
        """Returns array of vertices on convex hull of grain geometry.

        Note that only vertexes from exterior boundary are used.

        """
        return np.array(self.shape.convex_hull.exterior.xy)

    @property
    def perimeter(self):
        """Returns perimeter of grain

        """
        return self.shape.length

    @property
    def ead(self):
        """Returns equal area diameter of grain

        """
        return 2 * np.sqrt(self.shape.area / np.pi)

    def plot(self):
        """Plot ``Grain`` geometry on figure.

        Note that plotted ellipse reflects actual shape method

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        hull = self.hull
        ax.plot(*hull, ls='--', c='g')
        ax.add_patch(PathPatch(PolygonPath(self.shape),
                     fc='blue', ec='#000000', alpha=0.5, zorder=2))
        ax.plot(*self.centroid, color='red', marker='o')
        R = np.linspace(0, 360, 361)
        cr, sr = deg.cos(R), deg.sin(R)
        cl, sl = deg.cos(self.lao), deg.sin(self.lao)
        xx = self.xc + self.la * cr * sl / 2 + self.sa * sr * cl / 2
        yy = self.yc + self.la * cr * cl / 2 - self.sa * sr * sl / 2
        ax.plot(xx, yy, color='green')
        ax.plot(xx[[0, 180]], yy[[0, 180]], color='green')
        ax.plot(xx[[90, 270]], yy[[90, 270]], color='green')
        ax.autoscale_view(None, True, True)
        plt.title('LAO:{g.lao:g} AR:{g.ar} ({g.shape_method})'.format(g=self))
        return ax

    def show(self):
        """Show plot of ``Grain`` objects.

        """
        self.plot()
        plt.show()

    ##################################################################
    # Grain smooth and simplify methods (should return Grain object) #
    ##################################################################
    def spline(self, **kwargs):
        """Spline based smoothing of grains.

        Keywords:
          densify: factor for geometry densification. Default 5

        """
        x, y = _spline_ring(*self.xy, densify=kwargs.get('densify', 5))
        holes = []
        for hole in self.interiors:
            xh, yh = _spline_ring(*np.array(hole.xy),
                                  densify=kwargs.get('densify', 5))
            holes.append(LinearRing(coordinates=np.c_[xh, yh]))
        shape = Polygon(shell=LinearRing(coordinates=np.c_[x, y]), holes=holes)
        if shape.is_valid:
            res = Grain(shape, self.name, self.fid)
        else:
            res = self
            print('Invalid shape produced during smoothing of grain FID={}'.format(self.fid))
        return res

    def chaikin(self, **kwargs):
        """Chaikin corner-cutting smoothing algorithm.

        Keywords:
          repeat: Number of repetitions. Default 4

        """
        x, y = _chaikin_ring(*self.xy, repeat=kwargs.get('repeat', 4))
        holes = []
        for hole in self.interiors:
            xh, yh = _chaikin_ring(*np.array(hole.xy),
                                   repeat=kwargs.get('repeat', 4))
            holes.append(LinearRing(coordinates=np.c_[xh, yh]))
        shape = Polygon(shell=LinearRing(coordinates=np.c_[x, y]), holes=holes)
        if shape.is_valid:
            res = Grain(shape, self.name, self.fid)
        else:
            res = self
            print('Invalid shape produced during smoothing of grain FID={}'.format(self.fid))
        return res

    def dp(self, **kwargs):
        """Douglas–Peucker simplification.

        Keywords:
          tolerance: All points in the simplified object will be within the
          tolerance distance of the original geometry. Default Auto

        """
        if 'tolerance' not in kwargs:
            x, y = self.xy
            i1 = np.arange(len(x) - 2)
            i2 = i1 + 2
            i0 = i1 + 1
            d = abs((y[i2] - y[i1])*x[i0] - (x[i2] - x[i1])*y[i0] + x[i2]*y[i1] - y[i2]*x[i1])/np.sqrt((y[i2]-y[i1])**2 + (x[i2]-x[i1])**2)
            tolerance = d.mean()
        shape = self.shape.simplify(tolerance=kwargs.get('tolerance', tolerance),
                                    preserve_topology=kwargs.get('preserve_topology', False))
        return Grain(shape, self.name, self.fid)

    def vw(self, **kwargs):
        """Visvalingam-Whyatt simplification.

        The Visvalingam-Whyatt algorithm eliminates points based on their
        effective area. A points effective area is defined as the change
        in total area of the polygon by adding or removing that point.

        Keywords:
          minarea: Allowed total area change after simplification.
          Default value is calculated as 10% of grain area.

        """
        x, y = _visvalingam_whyatt_ring(*self.xy, minarea=kwargs.get('minarea', 0.01*self.area))
        holes = []
        for hole in self.interiors:
            xh, yh = _visvalingam_whyatt_ring(*np.array(hole.xy), minarea=kwargs.get('minarea', 0.01*Polygon(hole).area))
            holes.append(LinearRing(coordinates=np.c_[xh, yh]))
        shape = Polygon(shell=LinearRing(coordinates=np.c_[x, y]), holes=holes)
        if shape.is_valid:
            res = Grain(shape, self.name, self.fid)
        else:
            res = self
            print('Invalid shape produced during smoothing of grain FID={}'.format(self.fid))
        return res

    ################################################################
    # Grain shape methods (should modify sa, la, sao, lao, xc, yc) #
    ################################################################
    def minferet(self):
        """`shape_method`: minferet

        Short axis is defined as the minimum caliper of the polygon.
        Long axis correspond to caliper orthogonal to short axis.
        Center coordinates are set to centroid.

        """
        xy = self.hull.T
        dxy = xy[1:] - xy[:-1]
        ang = (deg.atan2(*dxy.T) + 90) % 180
        pp = np.dot(xy, np.array([deg.sin(ang), deg.cos(ang)]))
        d = pp.max(axis=0) - pp.min(axis=0)
        self.sa = np.min(d)
        self.sao = ang[d.argmin()]
        self.lao = (self.sao + 90) % 180
        self.la = self.feret(self.lao)
        self.xc, self.yc = self.shape.centroid.coords[0]
        self._shape_method = 'minferet'

    def moment(self):
        """`shape_method`: moment

        Short and long axes are calculated from area moments of inertia.
        Center coordinates are set to centroid. If moment fitting failed
        silently fallback to maxferet.

        """
        x, y = self.xy[:, :-1]
        x = x - x.mean()
        y = y - y.mean()
        xl = np.roll(x, -1)
        yl = np.roll(y, -1)
        v = xl * y - x * yl
        A = round(1e14 * np.sum(v) / 2) / 1e14

        if A != 0:
            a10 = np.sum(v * (xl + x)) / (6 * A)
            a01 = np.sum(v * (yl + y)) / (6 * A)
            a20 = np.sum(v * (xl**2 + xl * x + x**2)) / (12 * A)
            a11 = np.sum(v * (2 * xl * yl + xl * y + x * yl + 2 * x * y)) / (24 * A)
            a02 = np.sum(v * (yl**2 + yl * y + y**2)) / (12 * A)

            m20 = a20 - a10**2
            m11 = a11 - a10 * a01
            m02 = a02 - a01**2

            CM = np.array([[m02, -m11], [-m11, m20]]) / (4 * (m20 * m02 - m11**2))
            evals, evecs = np.linalg.eig(CM)
            idx = evals.argsort()
            evals = evals[idx]
            evecs = evecs[:, idx]
            self.la, self.sa = 2 / np.sqrt(evals)
            self.lao, self.sao = np.mod(deg.atan2(evecs[0, :], evecs[1, :]), 180)
            self.xc, self.yc = self.shape.centroid.coords[0]
            self._shape_method = 'moment'
        else:
            print('Moment fit failed for grain fid={}. Fallback to maxferet.'.format(self.fid))
            self.maxferet()

    def direct(self):
        """`shape_method`: direct

        Short, long axes and centre coordinates are calculated from direct
        least-square ellipse fitting. If direct fitting is not possible
        silently fallback to moment.

        """
        x, y = self.xy
        res = find_ellipse(x[:-1].copy(), y[:-1].copy())
        err = 1
        mx = 0
        while ((err > 1e-8) or np.isnan(err)) and (mx < 10):
            x, y = densify(x, y)
            res1 = find_ellipse(x[:-1].copy(), y[:-1].copy())
            err = np.sum((np.array(res1) - np.array(res))**2)
            res = res1
            mx += 1
        if mx == 10:
            print('Direct ellipse fit failed for grain fid={}. Fallback to moment.'.format(self.fid))
            self.moment()
        else:
            xc, yc, phi, a, b = res
            if a > b:
                ori = np.pi / 2 - phi
            else:
                ori = -phi
                a, b = b, a
            self.xc, self.yc, self.la, self.sa, self.lao, self.sao = xc, yc, 2*a, 2*b, np.rad2deg(ori) % 180, (np.rad2deg(ori) + 90) % 180
            self._shape_method = 'direct'

    def cov(self):
        """`shape_method`: cov

        Short and long axes are calculated from eigenvalue analysis
        of coordinate covariance matrix.

        """
        x, y = self.xy[:, :-1]
        x = x - x.mean()
        y = y - y.mean()
        s = np.cov(x, y)
        evals, evecs = np.linalg.eig(s)
        idx = evals.argsort()
        evals = evals[idx]
        evecs = evecs[:, idx]
        self.sa, self.la = np.sqrt(8) * np.sqrt(evals)
        self.sao, self.lao = np.mod(deg.atan2(evecs[0, :], evecs[1, :]), 180)
        self.xc, self.yc = self.shape.centroid.coords[0]
        self._shape_method = 'cov'


class Boundary(PolyShape):
    """Boundary class to store polyline boundary geometry

    A two-dimensional linear ring.

    """
    def __init__(self, shape, name='None-None', fid=0):
        """Create ``Boundary`` object

        """
        super(Boundary, self).__init__(shape, name, fid)
        self.shape_method = 'maxferet'

    def __repr__(self):
        return ('Boundary {b.fid:d} [{b.name:s}] '
                'L:{b.length:g}, AR:{b.ar:g}, '
                'LAO:{b.lao:g} ({b.shape_method:s})').format(b=self)

    @property
    def xy(self):
        """Returns array of vertex coordinate pair.

        """
        return np.array(self.shape.xy)

    @property
    def hull(self):
        """Returns array of vertices on convex hull of boundary geometry.

        """
        h = self.shape.convex_hull
        if h.geom_type == 'LineString':
            return np.array(h.xy)[:, [0, 1, 0]]
        else:
            return np.array(h.exterior.xy)

    def plot(self):
        """View ``Boundary`` geometry on figure.

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        hull = self.hull
        ax.plot(*hull, ls='--', c='g')
        ax.plot(*self.xy, c='blue')
        pa = np.array(list(itertools.combinations(range(len(hull.T)), 2)))
        d2 = np.sum((hull.T[pa[:, 0]] - hull.T[pa[:, 1]])**2, axis=1)
        ix = d2.argmax()
        ax.plot(*hull.T[pa[ix]].T, ls=':', lw=2, c='r')
        ax.autoscale_view(None, True, True)
        plt.title('LAO:{b.lao:g} AR:{b.ar} ({b.shape_method})'.format(b=self))
        return ax

    def show(self):
        """Show plot of ``Boundary`` objects.

        """
        self.plot()
        plt.show()

    ###################################################################
    # Boundary shape methods (should modify sa, la, sao, lao, xc, yc) #
    ###################################################################
    def cov(self):
        """`shape_method`: cov

        Short and long axes are calculated from eigenvalue analysis
        of coordinate covariance matrix.

        """
        x, y = self.xy
        x = x - x.mean()
        y = y - y.mean()
        s = np.cov(x, y)
        evals, evecs = np.linalg.eig(s)
        idx = evals.argsort()
        evals = evals[idx]
        evecs = evecs[:, idx]
        self.sa, self.la = np.sqrt(2) * np.sqrt(evals)
        self.sa = fixzero(self.sa)
        self.sao, self.lao = np.mod(deg.atan2(evecs[0, :], evecs[1, :]), 180)
        self.xc, self.yc = self.shape.centroid.coords[0]
        self._shape_method = 'cov'


class PolySet(object):
    """Base class to store set of ``Grains`` or ``Boundaries`` objects

    Properties:
      polys: list of objects
      extent: tuple of (xmin, ymin, xmax, ymax)

    """
    def __init__(self, shapes, attr='name', rule='unique', k=5):
        if len(shapes) > 0:
            self.polys = shapes
            gb = self.bounds
            self.extent = gb[:, 0].min(), gb[:, 1].min(), gb[:, 2].max(), gb[:, 3].max()
            self.classify(attr, rule=rule, k=k)
        else:
            raise ValueError("No objects passed.")

    def __iter__(self):
        return self.polys.__iter__()

    def __len__(self):
        return len(self.polys)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __contains__(self, v):
        return v in self.polys

    def __getattr__(self, attr):
        if attr == 'shape':
            res = np.array([getattr(p, attr) for p in self], dtype=object)
        else:
            try:
                res = np.array([getattr(p, attr) for p in self])
            except ValueError:
                res = [getattr(p, attr) for p in self]
        return res

    @property
    def shape_method(self):
        """Set or returns shape methods of all objects.

        """
        return [p.shape_method for p in self]

    @shape_method.setter
    def shape_method(self, value):
        for p in self:
            if not hasattr(p, '_shape_method'):
                p.shape_method = value
            if p._shape_method != value:
                p.shape_method = value

    def bootstrap(self, num=100, size=None):
        if size is None:
            size = len(self)
        for i in range(num):
            yield self[np.random.choice(len(self), size)]

    @property
    def width(self):
        """Returns width of extent.

        """
        return self.extent[2] - self.extent[0]

    @property
    def height(self):
        """Returns height of extent.

        """
        return self.extent[3] - self.extent[1]

    def feret(self, angle=0):
        """Returns array of feret diameters.

        Args:
            angle: Caliper angle. Default 0

        """
        return np.array([p.feret(angle) for p in self])

    def classify(self, attr, rule='natural', k=5):
        """Define classification of objects.

        Args:
          attr: name of attribute used for classification
          rule: type of classification
            'unique': unique value mapping (for discrete values)
            'equal': k equaly spaced bins (for continuos values)
            'user': bins edges defined by array k (for continuos values)
            'natural': natural breaks. Default rule
            'jenks': fischer jenks scheme

        Examples:
          >>> g.classify('name', 'unique')

        """
        self.class_attr = attr
        self.classes = Classify(getattr(self, attr), rule, k)

    def df(self, *attrs):
        """Returns ``pandas.DataFrame`` of object attributes.

        Example:
          >>> g.df('ead', 'ar')

        """
        attrs = list(attrs)
        if 'classes' in attrs:
            attrs[attrs.index('classes')] = 'class'
        if 'class' in attrs:
            attrs.remove('class')
            d = pd.DataFrame({self.class_attr + '_class': self.classes.names})
        else:
            d = pd.DataFrame()
        for attr in attrs:
            d[attr] = getattr(self, attr)
        return d

    def agg(self, *pairs):
        pieces = []
        for aggfunc, attr in zip(pairs[0::2], pairs[1::2]):
            df = getattr(self.groups(attr), aggfunc)()
            df.columns = ['{}_{}'.format(aggfunc, attr)]
            pieces.append(df)
        return pd.concat(pieces, axis=1).reindex(self.classes.index)

    def groups(self, *attrs):
        """Returns ``pandas.GroupBy`` of object attributes.

        Note that grouping is based on actual classification.

        Example:
          >>> g.classify('ar', 'natural')
          >>> g.groups('ead').mean()
                                    ead
              1.01765-1.31807  0.067772
              1.31807-1.54201  0.076206
              1.54201-1.82242  0.065400
              1.82242-2.36773  0.073690
              2.36773-12.1571  0.084016

        """
        df = self.df(*attrs)
        return df.groupby(self.classes.names)

    def _autocolortable(self, cmap='jet'):
        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
        n = len(self.classes.index)
        if n > 1:
            pos = np.round(np.linspace(0, cmap.N - 1, n))
        else:
            pos = [127]
        return dict(zip(self.classes.index, [cmap(int(i)) for i in pos]))

    def _makelegend(self, ax, pos='auto', ncol=3):
        if pos == 'auto':
            if self.width > self.height:
                pos = 'top'
                ncol = 3
            else:
                pos = 'right'
                ncol = 1

        if pos == 'top':
            h, l = ax.get_legend_handles_labels()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('top',
                                      size=0.25 + 0.25 * np.ceil(len(h) / ncol))
            cax.set_axis_off()
            cax.legend(h, l, loc=9, borderaxespad=0.,
                       ncol=ncol, bbox_to_anchor=[0.5, 1.1])
            plt.tight_layout()
        elif pos == 'right':
            h, l = ax.get_legend_handles_labels()
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=0.2 + 1.6 * ncol)
            cax.set_axis_off()
            cax.legend(h, l, loc=7, borderaxespad=0.,
                       bbox_to_anchor=[1.04, 0.5])
            plt.tight_layout()

    def plot(self, legend='auto', pos='auto', alpha=0.8,
             cmap='jet', ncol=1, show_fid=False, show_index=False):
        """Plot set of ``Grains`` or ``Boundaries`` objects.

        Args:
          legend: dictionary with classes as keys and RGB tuples as values
                  Default "auto" (created by _autocolortable method)
          pos: legend position "top", "right" or "none". Defalt "auto"
          alpha: transparency. Default 0.8
          cmap: colormap. Default "jet"
          ncol: number of columns for legend.

        Returns matplotlib axes object.

        """
        if legend == 'auto':
            legend = self._autocolortable(cmap)
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        self._plot(ax, legend, alpha)
        if show_index:
            for idx, p in enumerate(self):
                ax.text(p.xc, p.yc, str(idx),
                        bbox=dict(facecolor='yellow', alpha=0.5))
        if show_fid:
            for p in self:
                ax.text(p.xc, p.yc, str(p.fid),
                        bbox=dict(facecolor='yellow', alpha=0.5))
        plt.setp(plt.yticks()[1], rotation=90)
        self._makelegend(ax, pos, ncol)
        return ax

    def show(self, **kwargs):
        """Show plot of ``Grains`` or ``Boundaries`` objects.

        """
        self.plot(**kwargs)
        plt.show()

    def savefig(self, filename='figure.png', legend=None, pos='auto',
                alpha=0.8, cmap='jet', dpi=150, ncol=1):
        """Save grains or boudaries plot to file.

        Args:
          filename: file to save figure. Default "figure.png"
          dpi: DPI of image. Default 150
          See `plot` for other kwargs

        """
        if legend is None:
            legend = self._autocolortable(cmap)
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        self._plot(ax, legend, alpha)
        plt.setp(plt.yticks()[1], rotation=90)
        self._makelegend(ax, pos, ncol)
        plt.savefig(filename, dpi=dpi)
        plt.close()

    def rose(self, ang=None, bins=36, scaled=True, weights=None,
             density=False, arrow=0.95, rwidth=1,
             pdf=False, kappa=250, **kwargs):
        if ang is None:
            ang = self.lao
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        if pdf:
            from scipy.stats import vonmises
            theta = np.linspace(-np.pi, np.pi, 1801)
            radii = np.zeros_like(theta)
            for a in ang:
                radii += vonmises.pdf(theta, kappa, loc=np.radians(a))
                radii += vonmises.pdf(theta, kappa, loc=np.radians(a + 180))

            radii /= len(ang)
        else:
            width = 360 / bins
            num, bin_edges = np.histogram(np.concatenate((ang, ang + 180)),
                                          bins=bins + 1,
                                          range=(-width / 2, 360 + width / 2),
                                          weights=weights, density=density)
            num[0] += num[-1]
            num = num[:-1]
            theta = []
            radii = []
            for cc, val in zip(np.arange(0, 360, width), num):
                theta.extend([cc - width / 2, cc - rwidth * width / 2, cc,
                              cc + rwidth * width / 2, cc + width / 2, ])
                radii.extend([0, val * arrow, val, val * arrow, 0])
            theta = np.deg2rad(theta)
        if scaled:
            radii = np.sqrt(radii)
        ax.fill(theta, radii, **kwargs)
        return ax


class Grains(PolySet):
    """Class to store set of ``Grains`` objects

    """
    def __repr__(self):
        return 'Set of %s grains.' % len(self.polys)

    def __add__(self, other):
        return Grains(self.polys + other.polys)

    def __getitem__(self, index):
        """Fancy Grains indexing.

        Grains could be indexed by several ways based on type of index.
          int: returns Grain defined by index position
          string: returns Grains with index name
          list, tuple or np.array of int: returns Grains by index positions
          np.array of bool: return Grains where index is True

        Examples:
          >>> g[10]
          Grain 10 [qtz] A:0.0186429, AR:1.45201, LAO:39.6622 (moment)
          >>> g['qtz']
          Set of 155 grains.
          >>> g[g.ar > 3]
          Set of 41 grains.
          >>> g[g.classes(0)]   #grains from class 0
          Set of 254 grains.

        """
        if isinstance(index, list) or isinstance(index, tuple):
            index = np.asarray(index)
        if isinstance(index, str):
            index = np.flatnonzero(self.name == index)
        if isinstance(index, np.ndarray):
            if index.dtype == 'bool':
                index = np.flatnonzero(index)
            return Grains([self.polys[fid] for fid in index],
                          self.class_attr,
                          self.classes.rule,
                          self.classes.k)
        else:
            return self.polys[index]

    @property
    def phase_list(self):
        """Returns list of unique Grain names.

        """
        return list(np.unique(self.name))

    def boundaries(self, T=None):
        """Create Boundaries from Grains.

        Example:
          >>> g = Grains.from_shp()
          >>> b = g.boundaries()

        """
        from shapely.ops import linemerge

        shapes = []
        lookup = {}
        if T is None:
            T = nx.Graph()
        G = nx.DiGraph()
        for fid, g in enumerate(self):
            # get phase and add to list and legend
            path = []
            for co in g.shape.exterior.coords:
                if co not in lookup:
                    lookup[co] = len(lookup)
                path.append(lookup[co])
            G.add_path(path, fid=fid, phase=g.name)
            for holes in g.shape.interiors:
                path = []
                for co in holes.coords:
                    if co not in lookup:
                        lookup[co] = len(lookup)
                    path.append(lookup[co])
                G.add_path(path, fid=fid, phase=g.name)
        # Create topology graph
        H = G.to_undirected(reciprocal=True)
        for edge in H.edges_iter():
            e1 = G.get_edge_data(edge[0], edge[1])
            e2 = G.get_edge_data(edge[1], edge[0])
            bt = '%s-%s' % tuple(sorted([e1['phase'], e2['phase']]))
            T.add_node(e1['fid'])
            T.add_node(e2['fid'])
            T.add_edge(e1['fid'], e2['fid'], type=bt, bids=[])
        # Create boundaries
        for edge in T.edges_iter():
            shared = self[edge[0]].intersection(self[edge[1]])
            edge_data = T.get_edge_data(edge[0], edge[1])
            if shared.geom_type == 'LineString':  # LineString cannot be merged
                shapes.append(Boundary(shared, edge_data['type'], len(shapes)))
            elif shared.geom_type == 'MultiLineString':  # common case
                shared = linemerge(shared)
                if shared.geom_type == 'LineString':  # single shared boundary
                    bid = len(shapes)
                    shapes.append(Boundary(shared,
                                           edge_data['type'],
                                           bid))
                    edge_data['bids'].append(bid)
                else:  # multiple shared boundary
                    for sub in list(shared):
                        bid = len(shapes)
                        shapes.append(Boundary(sub,
                                               edge_data['type'],
                                               bid))
                        edge_data['bids'].append(bid)
            elif shared.geom_type == 'GeometryCollection':  # other cases
                for sub in shared:
                    if sub.geom_type == 'LineString':
                        bid = len(shapes)
                        shapes.append(Boundary(sub,
                                               edge_data['type'],
                                               bid))
                        edge_data['bids'].append(bid)
                    elif sub.geom_type == 'MultiLineString':
                        sub = linemerge(sub)
                        if sub.geom_type == 'LineString':
                            bid = len(shapes)
                            shapes.append(Boundary(sub,
                                                   edge_data['type'],
                                                   bid))
                            edge_data['bids'].append(bid)
                        else:
                            for subsub in list(sub):
                                bid = len(shapes)
                                shapes.append(Boundary(subsub,
                                                       edge_data['type'],
                                                       bid))
                                edge_data['bids'].append(bid)
        if not shapes:
            print('No shared boundaries found.')
        else:
            return Boundaries(shapes)

    def boundary_segments(self):
        """Create Boundaries from Grains boundary segments.

        Example:
          >>> g = Grains.from_shp()
          >>> b = g.boundary_segments()

        """
        from shapely.geometry import LineString

        shapes = []
        for g in self:
            for p0, p1 in zip(g.xy.T[:-1], g.xy.T[1:]):
                shapes.append(Boundary(LineString([p0, p1]), g.name, len(shapes)))
        return Boundaries(shapes)

    @classmethod
    def from_shp(cls, filename=os.path.join(respath, 'sg2.shp'),
                 phasefield='phase', phase='None'):
        """Create Grains from ESRI shapefile.

        Args:
          filename: filename of shapefile. Default sg2.shp from examples
          phasefield: name of attribute in shapefile that
            holds names of grains or None. Default "phase".
          phase: value used for grain phase when phasefield is None

        """
        sf = Reader(filename)
        if sf.shapeType == 5:
            fieldnames = [field[0].lower() for field in sf.fields[1:]]
            if phasefield is not None:
                if phasefield in fieldnames:
                    phase_pos = fieldnames.index(phasefield)
                else:
                    raise Exception("There is no field '%s'. Available fields are: %s" % (phasefield, fieldnames))
            shapeRecs = sf.shapeRecords()
            shapes = []
            for pos, rec in enumerate(shapeRecs):
                # A valid polygon must have at least 4 coordinate tuples
                if len(rec.shape.points) > 3:
                    geom = shape(rec.shape.__geo_interface__)
                    # try  to "clean" self-touching or self-crossing polygons
                    if not geom.is_valid:
                        geom = geom.buffer(0)
                    if geom.is_valid:
                        if not geom.is_empty:
                            if phasefield is None:
                                ph = phase
                            else:
                                ph = rec.record[phase_pos]
                            if geom.geom_type == 'MultiPolygon':
                                for g in geom:
                                    shapes.append(Grain(g, ph, len(shapes)))
                                print('Multipolygon (FID={}) exploded.'.format(pos))
                            elif geom.geom_type == 'Polygon':
                                shapes.append(Grain(geom, ph, len(shapes)))
                            else:
                                raise Exception('Unexpected geometry type (FID={})!'.format(pos))
                        else:
                            print('Empty geometry (FID={}) skipped.'.format(pos))
                    else:
                        print('Invalid geometry (FID={}) skipped.'.format(pos))
                else:
                    print('Invalid geometry (FID={}) skipped.'.format(pos))
            return cls(shapes)
        else:
            raise Exception('Shapefile must contains polygons!')

    def _plot(self, ax, legend, alpha, ec='#222222'):
        groups = self.groups('shape')
        keys = groups.groups.keys()
        for key in self.classes.index:
            paths = []
            if key in keys:
                group = groups.get_group(key)
                for g in group['shape']:
                    paths.append(PolygonPath(g))
                patch = PathPatch(Path.make_compound_path(*paths),
                                  fc=legend[key],
                                  ec=ec, alpha=alpha, zorder=2,
                                  label='{} ({})'.format(key, len(group)))
            else:
                patch = PathPatch(Path([[None, None]]),
                                  fc=legend[key],
                                  ec=ec, alpha=alpha, zorder=2,
                                  label='{} ({})'.format(key, 0))
            ax.add_patch(patch)
        ax.margins(0.025, 0.025)
        ax.get_yaxis().set_tick_params(which='both', direction='out')
        ax.get_xaxis().set_tick_params(which='both', direction='out')
        return ax

    def simplify(self, method='vw', **kwargs):
        return Grains([getattr(s, method)(**kwargs) for s in self])

    def smooth(self, method='chaikin', **kwargs):
        return Grains([getattr(s, method)(**kwargs) for s in self])


class Boundaries(PolySet):
    """Class to store set of ``Boundaries`` objects

    """
    def __repr__(self):
        return 'Set of %s boundaries.' % len(self.polys)

    def __add__(self, other):
        return Boundaries(self.polys + other.polys)

    def __getitem__(self, index):
        """Fancy Boundaries indexing.

        Boundaries could be indexed by several ways based on type of index.
          int: returns Boundary defined by index position
          string: returns Boundaries with index name (hyphen separated)
          list, tuple or np.array of int: returns Boundaries by index positions
          np.array of bool: return Grains where index is True

        Examples:
          >>> b[10]
          Boundary 10 [qtz-qtz] L:0.0982331, AR:1.41954, LAO:109.179 (maxferet)
          >>> b['qtz-pl']
          Set of 238 boundaries.
          >>> b[b.ar > 10]
          Set of 577 boundaries.
          >>> b[b.classes(0)]   #boundaries from class 0
          Set of 374 boundaries.

        """
        if isinstance(index, list) or isinstance(index, tuple):
            index = np.asarray(index)
        if isinstance(index, str):
            index = np.flatnonzero(self.name == index)
        if isinstance(index, np.ndarray):
            if index.dtype == 'bool':
                index = np.flatnonzero(index)
            return Boundaries([self.polys[fid] for fid in index],
                              self.class_attr,
                              self.classes.rule,
                              self.classes.k)
        else:
            return self.polys[index]

    @property
    def type_list(self):
        """Returns list of unique Boundary names.

        """
        return list(np.unique(self.name))

    def _plot(self, ax, legend, alpha):
        groups = self.groups('shape')
        for key in self.classes.index:
            group = groups.get_group(key)
            x = []
            y = []
            for b in group['shape']:
                xb, yb = b.xy
                x.extend(xb)
                x.append(np.nan)
                y.extend(yb)
                y.append(np.nan)
            ax.plot(x, y, color=legend[key], alpha=alpha, label='{} ({})'.format(key, len(group)))
        ax.margins(0.025, 0.025)
        ax.get_yaxis().set_tick_params(which='both', direction='out')
        ax.get_xaxis().set_tick_params(which='both', direction='out')
        return ax


class Sample(object):
    """Class to store both ``Grains`` and ``Boundaries`` objects

    Properties:
      g: Grains object
      b: Boundaries.objects
      T. ``networkx.Graph`` storing grain topology

    """
    def __init__(self):
        self.g = None
        self.b = None
        self.T = None

    def __repr__(self):
        return 'Sample with %s grains and %s boundaries.' % (len(self.g.polys),len(self.b.polys))

    @classmethod
    def from_shp(cls, filename=os.path.join(respath, 'sg2.shp'),
                 phasefield='phase'):
        return cls.from_grains(Grains.from_shp(filename, phasefield))

    @classmethod
    def from_grains(cls, grains):
        obj = cls()
        obj.T = nx.Graph()
        obj.g = grains
        obj.b = obj.g.boundaries(obj.T)
        return obj

    def neighbors(self, idx, name=None):
        """Returns array of indexes of neighbouring grains.

        If name attribute is provided only neighbours with name are returned

        """
        n = np.asarray(self.T.neighbors(idx))
        if name:
            n = n[self.g[n].name == name]
        return n

    def plot(self, legend=None, pos='auto', alpha=0.8,
             cmap='jet', ncol=1, show_fid=False, show_index=False):
        """Plot overlay of ``Grains`` and ``Boundaries`` of ``Sample`` object.

        Args:
          legend: dictionary with classes as keys and RGB tuples as values
                  Default Auto (created by _autocolortable method)
          pos: legend position "top" or "right". Defalt Auto
          alpha: transparency. Default 0.8
          cmap: colormap. Default "jet"
          ncol: number of columns for legend.

        Returns matplotlib axes object.

        """
        if legend is None:
            legend = dict(list(self.g._autocolortable().items()) + list(self.b._autocolortable().items()))
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        self.g._plot(ax, legend, alpha, ec='none')
        if show_index:
            for idx, p in enumerate(self.g):
                ax.text(p.xc, p.yc, str(idx),
                        bbox=dict(facecolor='yellow', alpha=0.5))
        if show_fid:
            for p in self.g:
                ax.text(p.xc, p.yc, str(p.fid),
                        bbox=dict(facecolor='yellow', alpha=0.5))
        self.b._plot(ax, legend, 1)
        if show_index:
            for idx, p in enumerate(self.b):
                ax.text(p.xc, p.yc, str(idx),
                        bbox=dict(facecolor='white', alpha=0.5))
        if show_fid:
            for p in self.b:
                ax.text(p.xc, p.yc, str(p.fid),
                        bbox=dict(facecolor='white', alpha=0.5))
        plt.setp(plt.yticks()[1], rotation=90)
        self.g._makelegend(ax, pos, ncol)
        return ax

    def show(self, **kwargs):
        """Show plot of ``Sample`` objects.

        """
        self.plot(**kwargs)
        plt.show()
