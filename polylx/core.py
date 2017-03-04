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
from shapely.geometry.polygon import orient
from shapely import affinity
import networkx as nx
import pandas as pd
import warnings

from .shapefile import Reader
from .utils import fixratio, fixzero, deg, Classify, PolygonPath
from .utils import find_ellipse, densify, inertia_moments
from .utils import _chaikin_ring, _spline_ring, _visvalingam_whyatt_ring

from pkg_resources import resource_filename

respath = resource_filename(__name__, 'example')
# ignore matplotlib deprecation warnings
warnings.filterwarnings('ignore', category=mcb.mplDeprecation)


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

    # def __getattr__(self, attr):
    #     if hasattr(self.shape, attr):
    #         return getattr(self.shape, attr)
    #     else:
    #         raise AttributeError

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
    def bounds(self):
        """Returns minimum bounding region (minx, miny, maxx, maxy)

        """
        return self.shape.bounds

    @property
    def area(self):
        """Area of the shape. For boundary returns 0.

        """
        return self.shape.area

    @property
    def length(self):
        """Unitless length of the geometry (float)

        """
        return self.shape.length

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

    @property
    def representative_point(self):
        """Returns a cheaply computed point that is guaranteed to be within the object.

        """
        return self.shape.representative_point().coords[0]

    def feret(self, angle=0):
        """Returns the ferret diameter for given angle.

        Args:
          angle: angle of caliper rotation

        """
        pp = np.dot(self.hull.T, np.array([deg.sin(angle), deg.cos(angle)]))
        return pp.max(axis=0) - pp.min(axis=0)

    def proj(self, angle=0):
        """Returns the cumulative projection of object for given angle.

        Args:
          angle: angle of projection line

        """
        pp = np.dot(self.xy.T, np.array([deg.sin(angle), deg.cos(angle)]))
        return abs(np.diff(pp, axis=0)).sum(axis=0)

    def surfor(self, angles=range(180), normalized=True):
        """Returns surfor function values. When normalized maximum value
        is 1 and correspond to max feret.

        Args:
          angles: iterable angle values. Defaut range(180)
          normalized: whether to normalize values. Defaut True

        """
        res = self.feret(angles)
        if normalized:
            #xy = self.hull.T
            #pa = np.array(list(itertools.combinations(range(len(xy)), 2)))
            #d2 = np.sum((xy[pa[:, 0]] - xy[pa[:, 1]])**2, axis=1)
            #res = res / np.sqrt(np.max(d2))
            res = res / res.max()
        return res

    def paror(self, angles=range(180), normalized=True):
        """Returns paror function values. When normalized maximum value
        is 1 and correspond to max feret.

        Args:
          angles: iterable angle values. Defaut range(180)
          normalized: whether to normalize values. Defaut True

        """
        res = self.proj(angles)
        if normalized:
            res = res / res.max()
        return res

    ##################################################################
    # Shapely/GEOS algorithms                                        #
    ##################################################################

    def contains(self, other):
        """Returns True if the geometry contains the other, else False

        """
        return self.shape.contains(other.shape)

    def crosses(self, other):
        """Returns True if the geometries cross, else False

        """
        return self.shape.crosses(other.shape)

    def difference(self, other):
        """Returns the difference of the geometries

        """
        return self.shape.difference(other.shape)

    def disjoint(self, other):
        """Returns True if geometries are disjoint, else False

        """
        return self.shape.disjoint(other.shape)

    def distance(self, other):
        """Unitless distance to other geometry (float)

        """
        return self.shape.distance(other.shape)

    def equals(self, other):
        """Returns True if geometries are equal, else False

        """
        return self.shape.equals(other.shape)

    def equals_exact(self, other, tolerance):
        """Returns True if geometries are equal to within a specified tolerance

        """
        return self.shape.equals_exact(other.shape, tolerance)

    def intersection(self, other):
        """Returns the intersection of the geometries

        """
        return self.shape.intersection(other.shape)

    def intersects(self, other):
        """Returns True if geometries intersect, else False

        """
        return self.shape.intersects(other.shape)

    def overlaps(self, other):
        """Returns True if geometries overlap, else False

        """
        return self.shape.overlaps(other.shape)

    def relate(self, other):
        """Returns the DE-9IM intersection matrix for the two geometries (string)

        """
        return self.shape.relate(other.shape)

    def symmetric_difference(self, other):
        """Returns the symmetric difference of the geometries (Shapely geometry)

        """
        return self.shape.symmetric_difference(other.shape)

    def touches(self, other):
        """Returns True if geometries touch, else False

        """
        return self.shape.relate(other.shape)

    def union(self, other):
        """Returns the union of the geometries (Shapely geometry)

        """
        return self.shape.union(other.shape)

    def within(self, other):
        """Returns True if geometry is within the other, else False

        """
        return self.shape.within(other.shape)

    ###################################################################
    # Shapely affinity methods                                        #
    ###################################################################

    def affine_transform(self, matrix):
        """Returns a transformed geometry using an affine transformation matrix.
        The matrix is provided as a list or tuple with 6 items:
        [a, b, d, e, xoff, yoff]
        which defines the equations for the transformed coordinates:
        x’ = a * x + b * y + xoff y’ = d * x + e * y + yoff

        """
        return type(self)(affinity.affine_transform(self.shape, matrix),
                          name=self.name, fid=self.fid)

    def rotate(self, angle, **kwargs):
        """Returns a rotated geometry on a 2D plane.
        The angle of rotation can be specified in either degrees (default)
        or radians by setting use_radians=True. Positive angles are
        counter-clockwise and negative are clockwise rotations.
        The point of origin can be a keyword ‘center’ for the object bounding
        box center (default), ‘centroid’ for the geometry’s centroid,
        or coordinate tuple (x0, y0) for fixed point.

        """
        return type(self)(affinity.rotate(self.shape, angle, **kwargs),
                          name=self.name, fid=self.fid)

    def scale(self, **kwargs):
        """Returns a scaled geometry, scaled by factors along each dimension.
        The point of origin can be a keyword ‘center’ for the object bounding
        box center (default), ‘centroid’ for the geometry’s centroid,
        or coordinate tuple (x0, y0) for fixed point.
        Negative scale factors will mirror or reflect coordinates.

        """
        return type(self)(affinity.scale(self.shape, **kwargs),
                          name=self.name, fid=self.fid)

    def skew(self, **kwargs):
        """Returns a skewed geometry, sheared by angles ‘xs’ along x and
        ‘ys’ along y direction. The shear angle can be specified in either
        degrees (default) or radians by setting use_radians=True.
        The point of origin can be a keyword ‘center’ for the object bounding
        box center (default), ‘centroid’ for the geometry’s centroid,
        or a coordinate tuple (x0, y0) for fixed point.

        """
        return type(self)(affinity.skew(self.shape, **kwargs),
                          name=self.name, fid=self.fid)

    def translate(self, **kwargs):
        """Returns a translated geometry shifted by offsets ‘xoff’ along x
        and ‘yoff’ along y direction.

        """
        return type(self)(affinity.translate(self.shape, **kwargs),
                          name=self.name, fid=self.fid)


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

    def copy(self):
        return Grain(self.shape, self.name, self.fid)

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
            return self(orient(geom), name, fid)
        else:
            print('Invalid geometry.')

    @property
    def xy(self):
        """Returns array of vertex coordinate pair.

        Note that only vertexes from exterior boundary are returned.
        For interiors use interiors property.

        """
        return np.array(self.shape.exterior.xy)

    @property
    def interiors(self):
        """Returns list of arrays of vertex coordinate pair of interiors.

        """
        return [np.array(hole.xy) for hole in self.shape.interiors]

    @property
    def hull(self):
        """Returns array of vertices on convex hull of grain geometry.

        """
        return np.array(self.shape.convex_hull.exterior.xy)

    @property
    def ead(self):
        """Returns equal area diameter of grain

        """
        return 2 * np.sqrt(self.area / np.pi)

    @property
    def nholes(self):
        """Returns number of holes (shape interiors)

        """
        return len(self.shape.interiors)

    def plot(self, **kwargs):
        """Plot ``Grain`` geometry on figure.

        Note that plotted ellipse reflects actual shape method

        """
        if 'ax' in kwargs:
            ax = kwargs['ax']
            ax.set_aspect('equal')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        hull = self.hull
        ax.plot(*hull, ls='--', c='green')
        ax.add_patch(PathPatch(PolygonPath(self.shape),
                     fc='blue', ec='#000000', alpha=0.5, zorder=2))
        ax.plot(*self.representative_point, color='coral', marker='o')
        ax.plot(*self.centroid, color='red', marker='o')
        ax.plot(self.xc, self.yc, color='green', marker='o')
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

    def show(self, **kwargs):
        """Show plot of ``Grain`` objects.

        """
        self.plot(**kwargs)
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
            xh, yh = _spline_ring(*hole,
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
            xh, yh = _chaikin_ring(*hole,
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
            d = (abs((y[i2] - y[i1]) * x[i0] -
                     (x[i2] - x[i1]) * y[i0] +
                     x[i2] * y[i1] - y[i2] * x[i1]) /
                 np.sqrt((y[i2] - y[i1]) ** 2 + (x[i2] - x[i1]) ** 2))
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
          Default value is calculated as 1% of grain area.

        """
        x, y = _visvalingam_whyatt_ring(*self.xy,
                                        minarea=kwargs.get('minarea', 0.01 * Polygon(self.exterior).area))
        holes = []
        for hole in self.interiors:
            xh, yh = _visvalingam_whyatt_ring(*hole,
                                              minarea=kwargs.get('minarea', 0.01 * Polygon(hole).area))
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
    def maxferet(self):
        """`shape_method`: maxferet

        Long axis is defined as the maximum caliper of the polygon.
        Short axis correspond to caliper orthogonal to long axis.
        Center coordinates are set to centroid of exterior.

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
        self.xc, self.yc = self.shape.exterior.centroid.coords[0]
        self._shape_method = 'maxferet'

    def minferet(self):
        """`shape_method`: minferet

        Short axis is defined as the minimum caliper of the polygon.
        Long axis correspond to caliper orthogonal to short axis.
        Center coordinates are set to centroid of exterior.

        """
        xy = self.hull.T
        dxy = xy[1:] - xy[:-1]
        ang = (deg.atan2(*dxy.T) + 90) % 180
        d = self.feret(ang)
        self.sa = np.min(d)
        self.sao = ang[d.argmin()]
        self.lao = (self.sao + 90) % 180
        self.la = self.feret(self.lao)
        self.xc, self.yc = self.shape.exterior.centroid.coords[0]
        self._shape_method = 'minferet'

    def minbox(self):
        """`shape_method`: minbox

        Short and long axes are claculated as widht and height of smallest
        area enclosing box.
        Center coordinates are set to centre of box.

        """
        xy = self.hull.T
        dxy = xy[1:] - xy[:-1]
        ang = (deg.atan2(*dxy.T)) % 180
        d1 = self.feret(ang)
        d2 = self.feret(ang + 90)
        ix = (d1 * d2).argmin()
        v1 = np.array([deg.sin(ang[ix]), deg.cos(ang[ix])])
        pp = np.dot(xy, v1)
        k1 = (pp.max() + pp.min()) / 2
        v2 = np.array([deg.sin(ang[ix] + 90), deg.cos(ang[ix] + 90)])
        pp = np.dot(xy, v2)
        k2 = (pp.max() + pp.min()) / 2
        self.xc, self.yc = k1 * v1 + k2 * v2
        if d1[ix] < d2[ix]:
            self.sa = d1[ix]
            self.sao = ang[ix]
            self.lao = (self.sao + 90) % 180
            self.la = d2[ix]
        else:
            self.sa = d2[ix]
            self.lao = ang[ix]
            self.sao = (self.lao + 90) % 180
            self.la = d1[ix]
        self._shape_method = 'minbox'

    def moment(self):
        """`shape_method`: moment

        Short and long axes are calculated from area moments of inertia.
        Center coordinates are set to centroid. If moment fitting failed
        silently fallback to maxferet.
        Center coordinates are set to centroid.

        """
        if not np.isclose(self.shape.area, 0):
            x, y = self.xy
            self.xc, self.yc = self.shape.centroid.coords[0]
            M = inertia_moments(x, y, self.xc, self.yc)
            for x, y in self.interiors:
                M -= inertia_moments(x, y, self.xc, self.yc)
            CM = np.array([[M[1], -M[2]], [-M[2], M[0]]]) / (4 * (M[0] * M[1] - M[2]**2))
            evals, evecs = np.linalg.eig(CM)
            idx = evals.argsort()
            evals = evals[idx]
            evecs = evecs[:, idx]
            self.la, self.sa = 2 / np.sqrt(evals)
            self.lao, self.sao = np.mod(deg.atan2(evecs[0, :], evecs[1, :]), 180)
            self.xc, self.yc = self.shape.centroid.coords[0]
            self._shape_method = 'moment'
        else:
            print('Moment fit failed for grain fid={} due to too small area. Fallback to maxferet.'.format(self.fid))
            self.maxferet()

    def direct(self):
        """`shape_method`: direct

        Short, long axes and centre coordinates are calculated from direct
        least-square ellipse fitting. If direct fitting is not possible
        silently fallback to moment.
        Center coordinates are set to centre of fitted ellipse.

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
        Center coordinates are set to centroid of exterior.

        """
        x, y = self.xy[:, :-1]
        self.xc, self.yc = self.shape.exterior.centroid.coords[0]
        s = np.cov(x - self.xc, y - self.yc)
        evals, evecs = np.linalg.eig(s)
        idx = evals.argsort()
        evals = evals[idx]
        evecs = evecs[:, idx]
        self.sa, self.la = np.sqrt(8) * np.sqrt(evals)
        self.sao, self.lao = np.mod(deg.atan2(evecs[0, :], evecs[1, :]), 180)
        self._shape_method = 'cov'

    def maee(self):
        """`shape_method`: maee

        Short and long axes are calculated from minimum volume enclosing
        ellipse. The solver is based on Khachiyan Algorithm, and the final
        solution is different from the optimal value by the pre-specified
        amount of tolerance of EAD/100.
        Center coordinates are set to centre of fitted ellipse.
        """
        P = self.hull[:, :-1]
        d, N = P.shape
        Q = np.vstack((P, np.ones(N)))
        count = 1
        err = 1
        u = np.ones(N) / N
        tol = self.ead / 100
        # Khachiyan Algorithm
        while err > tol:
            X = Q @ np.diag(u) @ Q.T
            M = np.diag(Q.T @ np.linalg.inv(X) @ Q)
            maximum, j = M.max(), M.argmax()
            step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
            new_u = (1 - step_size) * u
            new_u[j] = new_u[j] + step_size
            count += 1
            err = np.linalg.norm(new_u - u)
            u = new_u
        U = np.diag(u)
        A = np.linalg.inv(P @ U @ P.T - np.outer(P @ u, P @ u) ) / d
        evals, evecs = np.linalg.eig(A)
        idx = evals.argsort()
        evals = evals[idx]
        evecs = evecs[:, idx]
        self.la, self.sa = 2 / np.sqrt(evals)
        self.lao, self.sao = np.mod(deg.atan2(evecs[0, :], evecs[1, :]), 180)
        self.xc, self.yc = P @ u
        self._shape_method = 'maee'


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

    def copy(self):
        return Boundary(self.shape, self.name, self.fid)

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

    def plot(self, **kwargs):
        """View ``Boundary`` geometry on figure.

        """
        if 'ax' in kwargs:
            ax = kwargs['ax']
            ax.set_aspect('equal')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        hull = self.hull
        ax.plot(*hull, ls='--', c='green')
        ax.plot(*self.xy, c='blue')
        pa = np.array(list(itertools.combinations(range(len(hull.T)), 2)))
        d2 = np.sum((hull.T[pa[:, 0]] - hull.T[pa[:, 1]])**2, axis=1)
        ix = d2.argmax()
        ax.plot(*hull.T[pa[ix]].T, ls=':', lw=2, c='r')
        ax.autoscale_view(None, True, True)
        plt.title('LAO:{b.lao:g} AR:{b.ar} ({b.shape_method})'.format(b=self))
        return ax

    def show(self, **kwargs):
        """Show plot of ``Boundary`` objects.

        """
        self.plot(**kwargs)
        plt.show()

    ###################################################################
    # Boundary shape methods (should modify sa, la, sao, lao, xc, yc) #
    ###################################################################
    def maxferet(self):
        """`shape_method`: maxferet

        Long axis is defined as the maximum caliper of the polyline.
        Short axis correspond to caliper orthogonal to long axis.
        Center coordinates are set to centroid of polyline.

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
            fids = self.fid
            if len(fids) != len(np.unique(fids)):
                for ix, s in enumerate(self.polys):
                    s.fid = ix
                #print('FIDs are not unique and have been automatically changed.')
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

    def __getitem__(self, index):
        """Fancy indexing.

        Grains and Boundaries could be indexed by several ways based on type of index.
          int: returns objects defined by index position
          string: returns objects with index name
          list, tuple or np.array of int: returns objects by index positions
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
          >>> b[10]
          Boundary 10 [qtz-qtz] L:0.0982331, AR:1.41954, LAO:109.179 (maxferet)
          >>> b['qtz-pl']
          Set of 238 boundaries.
          >>> b[b.ar > 10]
          Set of 577 boundaries.
          >>> b[b.classes(0)]   #boundaries from class 0
          Set of 374 boundaries.

        """
        if isinstance(index, str):
            index = [i for i, n in enumerate(self.name) if n == index]
        if isinstance(index, list) or isinstance(index, tuple):
            index = np.asarray(index)
        if isinstance(index, slice):
            index = np.arange(len(self))[index]
        if isinstance(index, np.ndarray):
            if index.dtype == 'bool':
                index = np.flatnonzero(index)
            return type(self)([self.polys[fid] for fid in index],
                              self.class_attr,
                              self.classes.rule,
                              self.classes.k)
        else:
            return self.polys[index]

    def __contains__(self, v):
        return v in self.polys

    def _ipython_key_completions_(self):
        """ IPython integration of Tab key completions
        """
        return self.names

    # def __getattr__(self, attr, *args, **kwargs):
    #     res = []
    #     ismine = False
    #     for p in self:
    #         r = getattr(p, attr)
    #         if callable(r):
    #             r = r(*args, **kwargs)
    #         if isinstance(r, PolyShape):
    #             ismine = True
    #         res.append(r)
    #     if ismine:
    #         res = type(self)(res)
    #     else:
    #         try:
    #             res = np.array([getattr(p, attr) for p in self])
    #         except:
    #             pass
    #     return res

    ###################################################################
    # Shapely affinity methods                                        #
    ###################################################################

    def affine_transform(self, matrix):
        """Returns a transformed geometry using an affine transformation matrix.
        The matrix is provided as a list or tuple with 6 items:
        [a, b, d, e, xoff, yoff]
        which defines the equations for the transformed coordinates:
        x’ = a * x + b * y + xoff y’ = d * x + e * y + yoff

        """
        return type(self)([e.affine_transform(matrix) for e in self])

    def rotate(self, angle, **kwargs):
        """Returns a rotated geometry on a 2D plane.
        The angle of rotation can be specified in either degrees (default)
        or radians by setting use_radians=True. Positive angles are
        counter-clockwise and negative are clockwise rotations.
        The point of origin can be a keyword ‘center’ for the object bounding
        box center (default), ‘centroid’ for the geometry’s centroid,
        or coordinate tuple (x0, y0) for fixed point.

        """
        return type(self)([e.rotate(angle, **kwargs) for e in self])

    def scale(self, **kwargs):
        """Returns a scaled geometry, scaled by factors along each dimension.
        The point of origin can be a keyword ‘center’ for the object bounding
        box center (default), ‘centroid’ for the geometry’s centroid,
        or coordinate tuple (x0, y0) for fixed point.
        Negative scale factors will mirror or reflect coordinates.

        """
        return type(self)([e.scale(**kwargs) for e in self])

    def skew(self, **kwargs):
        """Returns a skewed geometry, sheared by angles ‘xs’ along x and
        ‘ys’ along y direction. The shear angle can be specified in either
        degrees (default) or radians by setting use_radians=True.
        The point of origin can be a keyword ‘center’ for the object bounding
        box center (default), ‘centroid’ for the geometry’s centroid,
        or a coordinate tuple (x0, y0) for fixed point.

        """
        return type(self)([e.skew(**kwargs) for e in self])

    def translate(self, **kwargs):
        """Returns a translated geometry shifted by offsets ‘xoff’ along x
        and ‘yoff’ along y direction.

        """
        return type(self)([e.translate(**kwargs) for e in self])

    ###################################################################
    # Shapely set-theoretic methods                                   #
    ###################################################################

    def clip(self, other):
        assert isinstance(other, Grain), 'Clipping is possible only by Grain.'
        res = []
        for e in self:
            if other.shape.intersects(e.shape):
                x = other.shape.intersection(e.shape)
                if x.geom_type == e.shape.geom_type:
                    res.append(type(e)(x, e.name, e.fid))
                elif x.geom_type == 'Multi' + e.shape.geom_type:
                    for xx in x:
                        res.append(type(e)(xx, e.name, e.fid))
                else:
                    pass
        return type(self)(res)

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
        """Bootstrap random sample generator.

        Args:
          num: number of boostraped samples. Default 100
          size: size of bootstraped samples. Default number of objects.

        Examples:
          >>> bsmean = np.mean([gs.ead.mean() for gs in g.bootstrap()])
        """
        if size is None:
            size = len(self)
        for i in range(num):
            yield self[np.random.choice(len(self), size)]

    def gridsplit(self, m=1, n=1):
        """Rectangular split generator.

        Args:
          m, n: number of rows and columns to split.

        Examples:
          >>> smean = np.mean([gs.ead.mean() for gs in g.gridsplit(6, 8)])
        """
        xmin, ymin, xmax, ymax = self.extent
        yoff = (ymax - ymin) / m
        xoff = (xmax - xmin) / n
        o = Grain.from_coords([xmin, xmin + xoff, xmin + xoff, xmin, xmin],
                              [ymin, ymin, ymin + yoff, ymin + yoff, ymin])

        for iy in range(m):
            for ix in range(n):
                c = o.translate(xoff=ix * xoff, yoff=iy * yoff)
                yield self.clip(c)

    def clipstrap(self, num=100, f=0.3):
        """Bootstrap random rectangular clip generator.

        Args:
          num: number of boostraped samples. Default 100
          f: area fraction clipped from original shape. Default 0.3

        Examples:
          >>> csmean = np.mean([gs.ead.mean() for gs in g.clipstrap()])
        """
        xmin, ymin, xmax, ymax = self.extent
        f = np.clip(f, 0, 1)
        w = f * (xmax - xmin)
        h = f * (ymax - ymin)
        for i in range(num):
            x = xmin + (1 - f) * (xmax - xmin) * np.random.random()
            y = ymin + (1 - f) * (ymax - ymin) * np.random.random()
            c = Grain.from_coords([x, x + w, x + w, x, x],
                                  [y, y, y + h, y + h, y])
            yield self.clip(c)

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

    @property
    def extent(self):
        """Returns minimum bounding region (minx, miny, maxx, maxy) of
        all objects

        """
        gb = np.array([p.bounds for p in self])
        return gb[:, 0].min(), gb[:, 1].min(), gb[:, 2].max(), gb[:, 3].max()

    @property
    def name(self):
        """Return list of names of the objects.

        """
        return [p.name for p in self]

    @property
    def names(self):
        """Returns list of unique object names.

        """
        return sorted(list(set(self.name)))

    @property
    def shape(self):
        """Return list of shapely objects.

        """
        return [p.shape for p in self]

    @property
    def la(self):
        """Return array of long axes of objects according to shape_method.

        """
        return np.array([p.la for p in self])

    @property
    def sa(self):
        """Return array of long axes of objects according to shape_method

        """
        return np.array([p.sa for p in self])

    @property
    def lao(self):
        """Return array of long axes of objects according to shape_method

        """
        return np.array([p.lao for p in self])

    @property
    def sao(self):
        """Return array of long axes of objects according to shape_method

        """
        return np.array([p.sao for p in self])

    @property
    def fid(self):
        """Return array of fids of objects.

        """
        return np.array([p.fid for p in self])

    def _fid(self, fid, first=True):
        """Return the indices of the objects with given fid.

        """
        ix = np.flatnonzero(self.fid == fid)
        if ix and first:
            return self[ix[0]]
        else:
            return self[ix]

    @property
    def area(self):
        """Return array of areas of the objects. For boundary returns 0.

        """
        return np.array([p.area for p in self])

    @property
    def length(self):
        """Return array of lengths of the objects.

        """
        return np.array([p.length for p in self])

    @property
    def ar(self):
        """Returns array of axial ratios

        Note that axial ratio is calculated from long and short axes
        calculated by actual ``shape method``.

        """
        return np.array([p.ar for p in self])

    @property
    def ma(self):
        """Returns mean axis

        Return array of mean axes calculated by actual ``shape method``.

        """
        return np.array([p.ma for p in self])

    @property
    def centroid(self):
        """Returns the 2D array of geometric centers of the objects

        """
        return np.array([p.centroid for p in self])

    @property
    def representative_point(self):
        """Returns a 2D array of cheaply computed points that are
        guaranteed to be within the objects.

        """
        return np.array([p.representative_point() for p in self])

    def feret(self, angle=0):
        """Returns array of feret diameters for given angle.

        Args:
            angle: Caliper angle. Default 0

        """
        return np.array([p.feret(angle) for p in self])

    def proj(self, angle=0):
        """Returns array of cumulative projection of object for given angle.
        Args:
          angle: angle of projection line

        """
        return np.array([p.proj(angle) for p in self])

    def surfor(self, angles=range(180), normalized=True):
        """Returns surfor function values. When normalized maximum value
        is 1 and correspond to max feret.

        Args:
          angles: iterable angle values. Defaut range(180)
          normalized: whether to normalize values. Defaut True

        """
        return np.array([p.surfor(angles, normalized) for p in self])

    def paror(self, angles=range(180), normalized=True):
        """Returns paror function values. When normalized maximum value
        is 1 and correspond to max feret.

        Args:
          angles: iterable angle values. Defaut range(180)
          normalized: whether to normalize values. Defaut True

        """
        return np.array([p.paror(angles, normalized) for p in self])

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
        idx = pd.Index(self.fid, name='fid')
        if 'class' in attrs:
            attrs.remove('class')
            d = pd.DataFrame({self.class_attr + '_class': self.classes.names}, index=idx)
        else:
            d = pd.DataFrame(index=idx)
        for attr in attrs:
            d[attr] = getattr(self, attr)
        return d

    def get(self, attr):
        """Returns ``pandas.Series`` of object attribute.

        Example:
          >>> g.get('ead')

        """
        idx = pd.Index(self.fid, name='fid')
        return pd.Series(getattr(self, attr), index=idx, name=attr)

    def agg(self, *pairs):
        """Returns concatenated result of multiple aggregations (different
        aggregation function for different attributes) based on actual
        classification. For single aggregation function use directly
        pandas groups, e.g. g.groups('lao', 'sao').agg(circular.mean)

        Example:
          >>> g.agg('area', np.sum, 'ead', np.mean, 'lao', circular.mean)
               sum_area  mean_ead  circular.mean_lao
          ksp  2.443733  0.089710          76.875574
          pl   1.083516  0.060629          94.331525
          qtz  1.166097  0.068071          74.318887

        """
        pieces = []
        for attr, aggfunc in zip(pairs[0::2], pairs[1::2]):
            df = self.groups(attr).agg(aggfunc)
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

    def nndist(self, **kwargs):
        from scipy.spatial import Delaunay
        pts = self.centroid
        tri = Delaunay(pts)
        T = nx.Graph()
        idx = np.arange(len(self))
        if kwargs.get('exclude_hull', True):
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            idx = np.setdiff1d(idx, hull.vertices)
        for i in idx:
            T.add_node(i)
            for n in tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i+1]]:
                T.add_node(n)
                T.add_edge(i, n)
        if kwargs.get('show', False):
            x = []
            y = []
            for e in T.edges():
                x += [pts[e[0]][0], pts[e[1]][0], np.nan]
                y += [pts[e[0]][1], pts[e[1]][1], np.nan]
            ax = self.plot()
            ax.plot(x, y, 'k')
            plt.show()
        return [np.sqrt(np.sum((pts[e[0]]-pts[e[1]])**2)) for e in T.edges()]

    def boundary_segments(self):
        """Create Boundaries from object boundary segments.

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

    def plot(self, **kwargs):
        """Plot set of ``Grains`` or ``Boundaries`` objects.

        Args:
          legend: dictionary with classes as keys and RGB tuples as values
                  Default "auto" (created by _autocolortable method)
          pos: legend position "top", "right" or "none". Defalt "auto"
          alpha: transparency. Default 0.8
          cmap: colormap. Default "jet"
          ncol: number of columns for legend.
          show_fid: Show FID of objects. Default False
          show_index: Show index of objects. Default False

        Returns matplotlib axes object.

        """
        if 'ax' in kwargs:
            ax = kwargs['ax']
            ax.set_aspect('equal')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        legend = kwargs.get('legend',
                            self._autocolortable(kwargs.get('cmap', 'jet')))
        self._plot(ax, legend, kwargs.get('alpha', 0.8))
        if kwargs.get('show_index', False):
            for idx, p in enumerate(self):
                ax.text(p.xc, p.yc, str(idx),
                        bbox=dict(facecolor='yellow', alpha=0.5))
        if kwargs.get('show_fid', False):
            for p in self:
                ax.text(p.xc, p.yc, str(p.fid),
                        bbox=dict(facecolor='yellow', alpha=0.5))
        plt.setp(plt.yticks()[1], rotation=90)
        self._makelegend(ax, kwargs.get('pos', 'auto'), kwargs.get('ncol', 1))
        return ax

    def show(self, **kwargs):
        """Show plot of ``Grains`` or ``Boundaries`` objects.

        """
        self.plot(**kwargs)
        plt.show()

    def savefig(self, **kwargs):
        """Save grains or boudaries plot to file.

        Args:
          filename: file to save figure. Default "figure.png"
          dpi: DPI of image. Default 150
          See `plot` for other kwargs

        """
        legend = kwargs.get('legend',
                            self._autocolortable(kwargs.get('cmap', 'jet')))
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        self._plot(ax, legend, kwargs.get('alpha', 0.8))
        plt.setp(plt.yticks()[1], rotation=90)
        self._makelegend(ax, kwargs.get('pos', 'auto'), kwargs.get('ncol', 1))
        plt.savefig(kwargs.get('filename', 'figure.png'),
                    dpi=kwargs.get('dpi', 150))
        plt.close()

    def rose(self, **kwargs):
        ang = kwargs.get('angles', self.lao)
        if 'ax' in kwargs:
            ax = kwargs.get('ax')
        else:
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
        if kwargs.get('scaled', True):
            radii = np.sqrt(radii)
        ax.fill(theta, radii, **kwargs.get('fill_kwg', {}))
        return ax


class Grains(PolySet):
    """Class to store set of ``Grains`` objects

    """
    def __repr__(self):
        #return 'Set of %s grains.' % len(self.polys)
        if len(self.names) == 1:
            res = 'Set of {:d} {:s} grains'.format(len(self), self.names[0])
        else:
            res = 'Set of {:d} grains with {:d} names'.format(len(self), len(self.names))
            if len(self.names) < 6:
                for p in self.names:
                    res += ' {:s}({:g})'.format(p, len(self[p]))
        return res

    def __add__(self, other):
        return Grains(self.polys + other.polys)

    @property
    def ead(self):
        """Returns array of equal area diameters of grains

        """
        return np.array([p.ead for p in self])

    @property
    def nholes(self):
        """Returns array of number of holes (shape interiors)

        """
        return np.array([p.nholes for p in self])

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
            bt = T[edge[0]][edge[1]]['type']
            bid = len(shapes)
            if shared.geom_type == 'LineString':  # LineString cannot be merged
                shapes.append(Boundary(shared, bt, bid))
                T[edge[0]][edge[1]]['bids'] = [bid]
            else:
                # Skip points if polygon just touch
                shared = linemerge([seg for seg in list(shared) if seg.geom_type is not 'Point'])
                if shared.geom_type == 'LineString':
                    shapes.append(Boundary(shared, bt, bid))
                    T[edge[0]][edge[1]]['bids'] = [bid]
                elif shared.geom_type == 'MultiLineString':
                    bids = []
                    for sub in list(shared):
                        bid = len(shapes)
                        shapes.append(Boundary(sub, bt, bid))
                        bids.append(bid)
                    T[edge[0]][edge[1]]['bids'] = bids
                else:
                    print('Upsss. Strange topology between polygons ', edge)
        if not shapes:
            print('No shared boundaries found.')
        else:
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
                                    shapes.append(Grain(orient(g), ph, len(shapes)))
                                print('Multipolygon (FID={}) exploded.'.format(pos))
                            elif geom.geom_type == 'Polygon':
                                shapes.append(Grain(orient(geom), ph, len(shapes)))
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
        #return 'Set of %s boundaries.' % len(self.polys)
        if len(self.names) == 1:
            res = 'Set of {:d} {:s} boundaries'.format(len(self), self.names[0])
        else:
            res = 'Set of {:d} boundaries with {:d} names'.format(len(self), len(self.names))
            if len(self.names) < 6:
                for p in self.names:
                    res += ' {:s}({:g})'.format(p, len(self[p]))
        return res

    def __add__(self, other):
        return Boundaries(self.polys + other.polys)

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
            ax.plot(x, y, color=legend[key], alpha=alpha,
                    label='{} ({})'.format(key, len(group)))
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
    def __init__(self, name=''):
        self.g = None
        self.b = None
        self.T = None
        self.name = name

    def __repr__(self):
        return 'Sample with %s grains and %s boundaries.' % (len(self.g.polys),
                                                             len(self.b.polys))

    @classmethod
    def from_shp(cls, filename=os.path.join(respath, 'sg2.shp'),
                 phasefield='phase', name=''):
        return cls.from_grains(Grains.from_shp(filename, phasefield), name=name)

    @classmethod
    def from_grains(cls, grains, name=''):
        obj = cls()
        obj.T = nx.Graph()
        obj.g = grains
        obj.b = obj.g.boundaries(obj.T)
        obj.name = name
        obj.pairs = {}
        for id1, id2 in obj.T.edges():
            for bid in obj.T[id1][id2]['bids']:
                obj.pairs[bid] = (id1, id2)
        return obj

    def neighbors(self, idx, name=None, inc=False):
        """Returns array of indexes of neighbouring grains.

        If name keyword is provided only neighbours with given name
        are returned.

        """
        try:
            idx = iter(idx)
        except TypeError:
            idx = iter([idx])
        res = set()
        for ix in idx:
            if ix in self.T:
                n = self.T.neighbors(ix)
                if name:
                    n = [i for i in n if self.g[i].name == name]
                res.update(n)
            if inc:
                res.add(ix)
        return list(res)

    def triplets(self):
        lookup = {}
        G = nx.Graph()
        for bid, b in enumerate(self.b):
            c0 = tuple(b.xy.T[0])
            c1 = tuple(b.xy.T[-1])
            if c0 not in lookup:
                lookup[c0] = len(lookup)
            if c1 not in lookup:
                lookup[c1] = len(lookup)
            G.add_edge(lookup[c0], lookup[c1], bid=bid)
        res = []
        for n0 in [n for n in G.degree() if G.degree()[n] == 3]:
            tri = set()
            for n1 in G.neighbors(n0):
                tri.update(bb[G[n0][n1]['bid']])
            res.append(tri)
        return res

    def bids(self, idx, name=None):
        nids = self.neighbors(idx, name=name)
        bids = []
        for nid in nids:
            bids.extend(self.T[idx][nid]['bids'])
        return bids

    def neighbors_dist(self, show=False, name=None):
        """Return array of nearest neighbors distances.

        If name keyword is provided only neighbours with given name
        are returned. When keyword show is True, plot is produced.

        """
        idx = self.T.nodes()
        pts = self.g.centroid
        if name:
            idx = [i for i in idx if self.g[i].name == name]
        T = nx.Graph()
        for i in idx:
            T.add_node(i)
            for n in self.neighbors(i, name=name):
                T.add_node(n)
                T.add_edge(i, n)
        if show:
            x = []
            y = []
            for e in T.edges():
                x += [pts[e[0]][0], pts[e[1]][0], np.nan]
                y += [pts[e[0]][1], pts[e[1]][1], np.nan]
            ax = self.g.plot()
            ax.plot(x, y, 'k')
            plt.show()
        return [np.sqrt(np.sum((pts[e[0]] - pts[e[1]]) ** 2))
                for e in T.edges()]

    def plot(self, **kwargs):
        """Plot overlay of ``Grains`` and ``Boundaries`` of ``Sample`` object.

        Args:
          legend: dictionary with classes as keys and RGB tuples as values
                  Default Auto (created by _autocolortable method)
          pos: legend position "top" or "right". Defalt Auto
          alpha: Grains transparency. Default 0.8
          gcmap: Grains colormap. Default "jet"
          bcmap: Boundary colormap. Default "jet"
          ncol: number of columns for legend.
          show_fid: Show FID of objects. Default False
          show_index: Show index of objects. Default False

        Returns matplotlib axes object.

        """
        if 'ax' in kwargs:
            ax = kwargs['ax']
            ax.set_aspect('equal')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, aspect='equal')
        legend = kwargs.get('legend',
                            dict(list(self.g._autocolortable(kwargs.get('gcmap', 'jet')).items()) +
                                 list(self.b._autocolortable(kwargs.get('bcmap', 'jet')).items())))
        alpha = kwargs.get('alpha', 0.8)
        show_fid = kwargs.get('show_fid', False)
        show_index = kwargs.get('show_index', False)
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
        self.g._makelegend(ax, kwargs.get('pos', 'auto'), kwargs.get('ncol', 1))
        ax.set_ylabel(self.name)
        return ax

    def show(self, **kwargs):
        """Show plot of ``Sample`` objects.

        """
        self.plot(**kwargs)
        plt.show()
