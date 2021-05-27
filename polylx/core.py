# -*- coding: utf-8 -*-
"""
Python module to visualize and analyze digitized 2D microstructures.

@author: Ondrej Lexa

Examples:
  >>> from polylx import *
  >>> g = Grains.from_shp()
  >>> b = g.boundaries()

"""
import os
import itertools
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import matplotlib.cbook as mcb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from shapely.geometry import shape, Polygon, LinearRing, LineString
from shapely.geometry.polygon import orient
from shapely import affinity
from shapely.ops import cascaded_union
import networkx as nx
import pandas as pd
import seaborn as sns
import warnings
import pyefd
from shapefile import Reader

try:
    import fiona
    fiona_OK = True
except ImportError:
    fiona_OK = False

from .utils import fixratio, fixzero, deg, Classify, PolygonPath
from .utils import find_ellipse, densify, inertia_moments
from .utils import _chaikin, _visvalingam_whyatt
from .utils import _spline_ring
from .utils import weighted_avg_and_std

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
        """Returns axial ratio (eccentricity)

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

    @property
    def pdist(self):
        """Returns a cummulative along-perimeter distances.

        """
        dxy = np.diff(self.xy, axis=1)
        dt = np.sqrt((dxy ** 2).sum(axis=0))
        return np.insert(np.cumsum(dt), 0, 0)

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
            # xy = self.hull.T
            # pa = np.array(list(itertools.combinations(range(len(xy)), 2)))
            # d2 = np.sum((xy[pa[:, 0]] - xy[pa[:, 1]])**2, axis=1)
            # res = res / np.sqrt(np.max(d2))
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

    def boundary_segments(self):
        """Create Boundaries from object boundary segments.

        Example:
          >>> g = Grains.from_shp()
          >>> b = g.boundaries()
          >>> bs1 = g[10].boundary_segments()
          >>> bs2 = b[10].boundary_segments()

        """
        shapes = []
        for p0, p1 in zip(self.xy.T[:-1], self.xy.T[1:]):
            shapes.append(Boundary(LineString([p0, p1]), self.name, len(shapes)))
        if isinstance(self, Grain):
            for hole in self.interiors:
                for p0, p1 in zip(hole.T[:-1], hole.T[1:]):
                    shapes.append(Boundary(LineString([p0, p1]), self.name, len(shapes)))
        return Boundaries(shapes)

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
        return self.shape.touches(other.shape)

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
        The point of origin can be a keyword 'center' for the object bounding
        box center (default), 'centroid' for the geometry’s centroid,
        or coordinate tuple (x0, y0) for fixed point.

        Args:
            angle
        Kwargs:
            origin='center'
            use_radians=False

        """
        return type(self)(affinity.rotate(self.shape, angle, **kwargs),
                          name=self.name, fid=self.fid)

    def scale(self, **kwargs):
        """Returns a scaled geometry, scaled by factors 'xfact' and 'yfact'
        along each dimension. The 'origin' keyword can be 'center' for the
        object bounding box center (default), 'centroid' for the geometry’s
        centroid, or coordinate tuple (x0, y0) for fixed point.
        Negative scale factors will mirror or reflect coordinates.

        Kwargs:
            xfact=1.0
            yfact=1.0
            origin='center'

        """
        return type(self)(affinity.scale(self.shape, **kwargs),
                          name=self.name, fid=self.fid)

    def skew(self, **kwargs):
        """Returns a skewed geometry, sheared by angles 'xs' along x and
        'ys' along y direction. The shear angle can be specified in either
        degrees (default) or radians by setting use_radians=True.
        The point of origin can be a keyword 'center' for the object bounding
        box center (default), 'centroid' for the geometry’s centroid,
        or a coordinate tuple (x0, y0) for fixed point.

        Kwrags:
            xs=0.0
            ys=0.0
            origin='center'
            use_radians=False

        """
        return type(self)(affinity.skew(self.shape, **kwargs),
                          name=self.name, fid=self.fid)

    def translate(self, **kwargs):
        """Returns a translated geometry shifted by offsets 'xoff' along x
        and 'yoff' along y direction.

        Kwargs:
            xoff=0.0
            yoff=0.0

        """
        return type(self)(affinity.translate(self.shape, **kwargs),
                          name=self.name, fid=self.fid)

    ###################################################################
    # Shapely affinity methods                                        #
    ###################################################################

    def dp(self, **kwargs):
        """Douglas–Peucker simplification.

        Keywords:
          tolerance: All points in the simplified object will be within the
          tolerance distance of the original geometry. Default Auto

        """
        x, y = self.xy
        if len(x) > 2:
            if 'tolerance' not in kwargs:
                i1 = np.arange(len(x) - 2)
                i2 = i1 + 2
                i0 = i1 + 1
                d = (abs((y[i2] - y[i1]) * x[i0] - (x[i2] - x[i1]) * y[i0] + x[i2] * y[i1] - y[i2] * x[i1]) / np.sqrt((y[i2] - y[i1]) ** 2 + (x[i2] - x[i1]) ** 2))
                tolerance = d.mean()
            shape = self.shape.simplify(kwargs.get('tolerance', tolerance), False)
            if shape.is_empty:
                shape = self.shape.simplify(kwargs.get('tolerance', tolerance), True)
            if shape.is_empty:
                shape = self.shape
                print('Invalid shape produced during smoothing for FID={}'.format(self.fid))
        else:
            shape = self.shape
        return type(self)(shape, self.name, self.fid)


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
    def circularity(self):
        """Return circularity (also called compactness) of the object.
        circ = length**2/ (4 * pi * area)

        """
        return self.length**2 / (4 * np.pi * self.area)

    @property
    def haralick(self):
        """Return Haralick’s circularity of the object.
        hcirc = mean(R) / std(R) where R is array of centroid-vertex distances

        """
        r = self.cdist
        return np.mean(r) / np.std(r)

    @property
    def cdist(self):
        """Returns centroid-vertex distances of grain exterior

        """
        return np.sqrt(np.sum((self.xy.T - self.centroid)**2, axis=1))

    @property
    def cdir(self):
        """Returns centroid-vertex directions of grain exterior

        """
        return np.arctan2(*(self.xy.T - self.centroid).T)

    def shape_vector(self, **kwargs):
        """Returns shape (feature) vector.

        Shape (feature) vector is calculated from Fourier descriptors (FD)
        to index the shape. To achieve rotation invariance, phase information
        of the FDs are ignored and only the magnitudes |FDn| are used. Scale
        invariance is achieved by dividing the magnitudes by the DC component,
        i.e., |FD0|. Since centroid distance is a real value function, only half
        of the FDs are needed to index the shape.

        Keywords:
          N: number of vertices to regularize outline. Default 128
             Note that number returned FDs is half of N.

        """
        N = kwargs.get('N', 128)
        r = self.regularize(N=N).cdist
        fft = np.fft.fft(r)
        f = abs(fft[1:]) / abs(fft[0])
        return f[:int(N / 2)]

    @property
    def nholes(self):
        """Returns number of holes (shape interiors)

        """
        return len(self.shape.interiors)

    def plot(self, **kwargs):
        """Plot ``Grain`` geometry on figure.

        Note that plotted ellipse reflects actual shape method

        """
        vertices = kwargs.get('vertices', False)
        if 'ax' in kwargs:
            ax = kwargs['ax']
            ax.set_aspect('equal')
        else:
            fig = plt.figure(figsize=kwargs.get('figsize', plt.rcParams.get('figure.figsize')))
            ax = fig.add_subplot(111, aspect='equal')
        hull = self.hull
        ax.plot(*hull, ls='--', c='green')
        ax.add_patch(PathPatch(PolygonPath(self.shape),
                     fc='blue', ec='#000000', alpha=0.5, zorder=2))
        if vertices:
            ax.plot(*self.xy, marker='.', c='blue')
            for hole in self.interiors:
                ax.plot(*hole, marker='.', c='blue')
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
        ax.set_title('LAO:{g.lao:g} AR:{g.ar} ({g.shape_method})'.format(g=self))
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
          repeat: Number of repetitions. Default 2

        """
        repeat = kwargs.get('repeat', 2)
        x, y = _chaikin(*self.xy, repeat=repeat, is_ring=True)
        holes = []
        for hole in self.interiors:
            xh, yh = _chaikin(*hole, repeat=repeat, is_ring=True)
            holes.append(LinearRing(coordinates=np.c_[xh, yh]))
        shape = Polygon(shell=LinearRing(coordinates=np.c_[x, y]), holes=holes)
        if shape.is_valid:
            res = Grain(shape, self.name, self.fid)
        else:
            res = self
            print('Invalid shape produced during smoothing of grain FID={}'.format(self.fid))
        return res

    def vw(self, **kwargs):
        """Visvalingam-Whyatt simplification.

        The Visvalingam-Whyatt algorithm eliminates points based on their
        effective area. A points effective area is defined as the change
        in total area of the polygon by adding or removing that point.

        Keywords:
          threshold: Allowed total boundary length change in percents. Default 1

        """
        threshold = kwargs.get('threshold', 1)
        x, y = _visvalingam_whyatt(*self.xy, threshold=threshold, is_ring=True)
        holes = []
        for hole in self.interiors:
            xh, yh = _visvalingam_whyatt(*hole, threshold=threshold, is_ring=True)
            holes.append(LinearRing(coordinates=np.c_[xh, yh]))
        shape = Polygon(LinearRing(coordinates=np.c_[x, y]), holes=holes)
        if shape.is_valid:
            res = Grain(shape, self.name, self.fid)
        else:
            res = self
            print('Invalid shape produced during smoothing of grain FID={}'.format(self.fid))
        return res

    def regularize(self, **kwargs):
        """Grain vertices regularization.

        Returns ``Grain`` object defined by vertices regularly distributed
        along boundaries of original ``Grain``.

        Keywords:
          N: Number of vertices. Default 128.
          length: approx. length of segments. Default None

        """
        N = kwargs.get('N', 128)
        if 'length' in kwargs:
            N = int(self.shape.exterior.length / kwargs['length']) + 1
            N = max(N, 4)
        rc = np.asarray([self.shape.exterior.interpolate(d, normalized=True).xy
                         for d in np.linspace(0, 1, N)])[:, :, 0]
        holes = []
        for hole in self.shape.interiors:
            if 'length' in kwargs:
                N = int(hole.length / kwargs['length']) + 1
                N = max(N, 4)
            rh = np.asarray([hole.interpolate(d, normalized=True).xy
                             for d in np.linspace(0, 1, N)])[:, :, 0]
            holes.append(LinearRing(rh))
        return Grain(Polygon(rc, holes=holes), self.name, self.fid)

    def fourier(self, **kwargs):
        """Eliptic Fourier reconstruction.

        Returns reconstructed ``Grain`` object using Fourier coefficients
        for characterizing closed contours.

        Keywords:
          order: The order of FDC to calculate. Default 12.
          N: number of vertices for reconstructed grain. Default 128.

        """
        order = kwargs.get('order', self.xy.shape[1])
        N = kwargs.get('N', 128)
        coeffs = pyefd.elliptic_fourier_descriptors(self.xy.T, order=order)
        locus = pyefd.calculate_dc_coefficients(self.xy.T)
        x, y = pyefd.reconstruct_contour(coeffs, locus=locus, num_points=N).T
        holes = []
        for hole in self.interiors:
            order = kwargs.get('order', hole.shape[1])
            coeffs = pyefd.elliptic_fourier_descriptors(hole.T, order=order)
            locus = pyefd.calculate_dc_coefficients(hole.T)
            xh, yh = pyefd.reconstruct_contour(coeffs, locus=locus, num_points=N).T
            holes.append(LinearRing(coordinates=np.c_[xh, yh]))
        shape = Polygon(LinearRing(coordinates=np.c_[x, y]), holes=holes)
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
            self.xc, self.yc, self.la, self.sa, self.lao, self.sao = xc, yc, 2 * a, 2 * b, np.rad2deg(ori) % 180, (np.rad2deg(ori) + 90) % 180
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

        Short and long axes are calculated from minimum area enclosing
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
            # X = Q @ np.diag(u) @ Q.T
            X = Q.dot(np.diag(u)).dot(Q.T)
            # M = np.diag(Q.T @ np.linalg.inv(X) @ Q)
            M = np.diag(Q.T.dot(np.linalg.inv(X)).dot(Q))
            maximum, j = M.max(), M.argmax()
            step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
            new_u = (1 - step_size) * u
            new_u[j] = new_u[j] + step_size
            count += 1
            err = np.linalg.norm(new_u - u)
            u = new_u
        U = np.diag(u)
        # A = np.linalg.inv(P @ U @ P.T - np.outer(P @ u, P @ u)) / d
        A = np.linalg.inv(P.dot(U).dot(P.T) - np.outer(P.dot(u), P.dot(u))) / d
        evals, evecs = np.linalg.eig(A)
        idx = evals.argsort()
        evals = evals[idx]
        evecs = evecs[:, idx]
        self.la, self.sa = 2 / np.sqrt(evals)
        self.lao, self.sao = np.mod(deg.atan2(evecs[0, :], evecs[1, :]), 180)
        self.xc, self.yc = P.dot(u)
        self._shape_method = 'maee'

    def fourier_ellipse(self):
        """`shape_method`: fourier_ellipse

        Short and long axes are calculated from first-order approximation
        of contour with a Fourier series.

        """
        coeffs = pyefd.elliptic_fourier_descriptors(self.xy.T, order=1)
        coeffs, psi = pyefd.normalize_efd(coeffs, size_invariant=False)
        self.xc, self.yc = pyefd.calculate_dc_coefficients(self.xy.T)
        self.sa, self.la = 2 * coeffs[0, [3, 0]]
        self.sao, self.lao = np.degrees(np.pi - psi) % 180, np.degrees(np.pi / 2 - psi) % 180
        self._shape_method = 'fourier_ellipse'


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
        vertices = kwargs.get('vertices', False)
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            ax.set_aspect('equal')
        else:
            fig = plt.figure(figsize=kwargs.get('figsize', plt.rcParams.get('figure.figsize')))
            ax = fig.add_subplot(111, aspect='equal')
        ax.plot(*self.xy, c='blue')
        if vertices:
            ax.plot(*self.xy, marker='.', c='blue')
        hull = self.hull
        ax.plot(*hull, ls='--', c='green')
        pa = np.array(list(itertools.combinations(range(len(hull.T)), 2)))
        d2 = np.sum((hull.T[pa[:, 0]] - hull.T[pa[:, 1]])**2, axis=1)
        ix = d2.argmax()
        ax.plot(*hull.T[pa[ix]].T, ls=':', lw=2, c='r')
        ax.autoscale_view(None, True, True)
        ax.set_title('LAO:{b.lao:g} AR:{b.ar} ({b.shape_method})'.format(b=self))
        return ax

    def show(self, **kwargs):
        """Show plot of ``Boundary`` objects.

        """
        self.plot(**kwargs)
        plt.show()

    ########################################################################
    # Boundary smooth and simplify methods (should return Boundary object) #
    ########################################################################
    # def spline(self, **kwargs):
    #     """Spline based smoothing of grains.

    #     Keywords:
    #       densify: factor for geometry densification. Default 5

    #     """
    #     x, y = _spline_ring(*self.xy, densify=kwargs.get('densify', 5))
    #     holes = []
    #     for hole in self.interiors:
    #         xh, yh = _spline_ring(*hole,
    #                               densify=kwargs.get('densify', 5))
    #         holes.append(LinearRing(coordinates=np.c_[xh, yh]))
    #     shape = Polygon(shell=LinearRing(coordinates=np.c_[x, y]), holes=holes)
    #     if shape.is_valid:
    #         res = Grain(shape, self.name, self.fid)
    #     else:
    #         res = self
    #         print('Invalid shape produced during smoothing of grain FID={}'.format(self.fid))
    #     return res

    def chaikin(self, **kwargs):
        """Chaikin corner-cutting smoothing algorithm.

        Keywords:
          repeat: Number of repetitions. Default 2

        """
        repeat = kwargs.get('repeat', 2)
        x, y = _chaikin(*self.xy, repeat=repeat, is_ring=self.shape.is_ring)
        shape = LineString(coordinates=np.c_[x, y])
        if shape.is_valid:
            res = Boundary(shape, self.name, self.fid)
        else:
            res = self
            print('Invalid shape produced during smoothing of boundary FID={}'.format(self.fid))
        return res

    def vw(self, **kwargs):
        """Visvalingam-Whyatt simplification.

        The Visvalingam-Whyatt algorithm eliminates points based on their
        effective area. A points effective area is defined as the change
        in total area of the polygon by adding or removing that point.

        Keywords:
          threshold: Allowed total boundary length change in percents. Default 1

        """
        threshold = kwargs.get('threshold', 1)
        x, y = _visvalingam_whyatt(*self.xy, threshold=threshold, is_ring=self.shape.is_ring)
        shape = LineString(np.c_[x, y])
        return Boundary(shape, self.name, self.fid)

    def regularize(self, **kwargs):
        """Boundary vertices regularization.

        Returns ``Boundary`` object defined by vertices regularly distributed
        along original ``Boundary``.

        Keywords:
          N: Number of vertices. Default 128.
          length: approx. length of segments. Default None

        """
        N = kwargs.get('N', 128)
        if 'length' in kwargs:
            N = int(self.length / kwargs['length']) + 1
            N = max(N, 2)
        rc = np.asarray([self.shape.interpolate(d, normalized=True).xy
                         for d in np.linspace(0, 1, N)])[:, :, 0]
        return Boundary(LineString(rc), self.name, self.fid)

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
    def __init__(self, shapes, classification=None):
        if len(shapes) > 0:
            self.polys = shapes
            fids = self.fid
            if len(fids) != len(np.unique(fids)):
                for ix, s in enumerate(self.polys):
                    s.fid = ix
                # print('FIDs are not unique and have been automatically changed.')
            if classification is None:
                self.classify('name', rule='unique')
            else:
                self.classes = classification
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
            if index.size > 0:
                if index.dtype == 'bool':
                    index = np.flatnonzero(index)
                return type(self)([self.polys[ix] for ix in index],
                                  self.classes[index])
            else:
                print('No result...')
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

    def clip(self, *bounds):
        """Clip by bounds rectangle (minx, miny, maxx, maxy) tuple (float values)
        """
        assert len(bounds) == 4, 'Bound must be defined by (minx, miny, maxx, maxy) tuple.'
        minx, miny, maxx, maxy = bounds
        return self.clip_by_shape(Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)]))

    def clip_by_shape(self, other):
        assert isinstance(other, Polygon), 'Clipping is possible only by shapely Polygon.'
        other = other.buffer(0)  # fix common problems
        res = []
        for e in self:
            if other.intersects(e.shape):
                x = other.intersection(e.shape)
                if x.geom_type == e.shape.geom_type:
                    res.append(type(e)(x, e.name, e.fid))
                elif x.geom_type == 'Multi' + e.shape.geom_type:
                    for xx in x:
                        res.append(type(e)(xx, e.name, e.fid))
                else:
                    pass
        if res:
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
        o = Polygon([(xmin, ymin), (xmin + xoff, ymin), (xmin + xoff, ymin + yoff), (xmin, ymin + yoff)])

        for iy in range(m):
            for ix in range(n):
                yield self.clip_by_shape(affinity.translate(o, xoff=ix * xoff, yoff=iy * yoff))

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
            yield self.clip_by_shape(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]))

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
    def features(self):
        """Generator of feature records

        """
        for p in self:
            feature = {'geometry': p.shape.__geo_interface__,
                       'properties': {'id': p.fid, 'name': p.name}}
            yield feature

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

    def getindex(self, name):
        """Return the indices of the objects with given name.

        """
        return [i for i, n in enumerate(self.name) if n == name]

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
        return np.array([p.representative_point for p in self])

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

    def classify(self, *args, **kwargs):
        """Define classification of objects.

        When no aruments are provided, default unique classification
        based on name attribute is used.

        Args:
          vals: name of attribute (str) used for classification
                or array of values
        Keywords:
          label: used as classification label when vals is array
          k: number of classes for continuous values
          rule: type of classification
            'unique': unique value mapping (for discrete values)
            'equal': k equaly spaced bins (for continuos values)
            'user': bins edges defined by array k (for continuos values)
            'natural': natural breaks. Default rule.
                       (beware not always unique solution)
            'jenks': fischer jenks scheme
          cmap: matplotlib colormap. Default 'viridis'

        Examples:
          >>> g.classify('name', rule='unique')
          >>> g.classify('ar', rule='jenks', k=5)

        """
        assert len(args) < 2, ('More than one argument passed...')
        if len(args) == 0:
            if 'rule' not in kwargs:
                kwargs['rule'] = 'unique'
            self.classify('name', **kwargs)
        else:
            vals = args[0]
            if isinstance(vals, str):
                if 'label' not in kwargs:
                    kwargs['label'] = vals
                self.classes = Classify(getattr(self, vals), **kwargs)
            else:
                if 'label' not in kwargs:
                    kwargs['label'] = 'User values'
                self.classes = Classify(vals, **kwargs)

    def get_class(self, key):
        assert key in self.class_names, ("Nonexisting class...")
        ix = self.class_names.index(key)
        return self[self.classes(ix)]

    def class_iter(self):
        for key in self.class_names:
            yield key, self.get_class(key)

    @property
    def class_names(self):
        return self.classes.index

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
            # d = pd.DataFrame({self.classes.label + '_class': self.classes.names}, index=idx)
            d = pd.DataFrame({'class': self.classes.names}, index=idx)
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
                     area       ead        lao
          class
          ksp    2.443733  0.089710  76.875488
          pl     1.083516  0.060629  94.197847
          qtz    1.166097  0.068071  74.320337

        """
        pieces = []
        for attr, aggfunc in zip(pairs[0::2], pairs[1::2]):
            df = self.groups(attr).agg(aggfunc)
            pieces.append(df)
        return pd.concat(pieces, axis=1).reindex(self.class_names)

    def accumulate(self, *methods):
        """Returns accumulated result of multiple Group methods based
        on actual classification.

        Example:
          >>> g.accumulate('rms_ead', 'aw_ead', 'aw_ead_log')
                  rms_ead    aw_ead  aw_ead_log
          class
          ksp    0.110679  0.185953    0.161449
          pl     0.068736  0.095300    0.086762
          qtz    0.097872  0.297476    0.210481

        """
        pieces = []
        for key, g in self.class_iter():
            row = {'class': key}
            for method in methods:
                row[method] = getattr(g, method)
            pieces.append(row)
        return pd.DataFrame(pieces).set_index('class')

    def groups(self, *attrs):
        """Returns ``pandas.GroupBy`` of object attributes.

        Note that grouping is based on actual classification.

        Example:
          >>> g.classify('ar', rule='natural')
          >>> g.groups('ead').mean()
                           ead
          class
          1.02-1.32   0.067772
          1.32-1.54   0.076042
          1.54-1.82   0.065479
          1.82-2.37   0.073690
          2.37-12.16  0.084016

        """
        df = self.df('class', *attrs)
        return df.groupby('class')

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
            for n in tri.vertex_neighbor_vertices[1][tri.vertex_neighbor_vertices[0][i]:tri.vertex_neighbor_vertices[0][i + 1]]:
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
        return [np.sqrt(np.sum((pts[e[0]] - pts[e[1]])**2)) for e in T.edges()]

    def boundary_segments(self):
        """Create Boundaries from object boundary segments.

        Example:
          >>> g = Grains.from_shp()
          >>> b = g.boundary_segments()

        """
        shapes = []
        for g in self:
            for p0, p1 in zip(g.xy.T[:-1], g.xy.T[1:]):
                shapes.append(Boundary(LineString([p0, p1]), g.name, len(shapes)))
            if isinstance(g, Grain):
                for hole in g.interiors:
                    for p0, p1 in zip(hole.T[:-1], hole.T[1:]):
                        shapes.append(Boundary(LineString([p0, p1]), g.name, len(shapes)))
        return Boundaries(shapes)

    def _makelegend(self, ax, **kwargs):
        pos = kwargs.get('pos', 'auto')
        ncol = kwargs.get('ncol', 3)
        if pos == 'auto':
            if self.width > self.height:
                pos = 'top'
            else:
                pos = 'right'
                ncol = kwargs.get('ncol', 1)

        if pos == 'top':
            h, lbls = ax.get_legend_handles_labels()
            if h:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('top',
                                          size=0.25 + 0.25 * np.ceil(len(h) / ncol))
                cax.set_axis_off()
                cax.legend(h, lbls, loc=9, borderaxespad=0.,
                           ncol=ncol, bbox_to_anchor=[0.5, 1.1])
            plt.tight_layout()
        elif pos == 'right':
            h, lbls = ax.get_legend_handles_labels()
            if h:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size=0.2 + 1.6 * ncol)
                cax.set_axis_off()
                cax.legend(h, lbls, loc=7, borderaxespad=0.,
                           bbox_to_anchor=[1.04, 0.5])
            plt.tight_layout()

    def plot(self, **kwargs):
        """Plot set of ``Grains`` or ``Boundaries`` objects.

        Keywords:
          alpha: transparency. Default 0.8
          pos: legend position "top", "right" or "none". Defalt "auto"
          ncol: number of columns for legend.
          legend: Show legend. Default True
          show_fid: Show FID of objects. Default False
          show_index: Show index of objects. Default False
          scalebar: When True scalebar is drawn instead axes frame
          scalebar_kwg: Dict of scalebar properties
            size: Default 1
            label: Default 1mm
            loc: Default 'lower right'
            See AnchoredSizeBar for others

        Returns matplotlib axes object.

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            ax.set_aspect('equal')
        else:
            fig = plt.figure(figsize=kwargs.get('figsize', plt.rcParams.get('figure.figsize')))
            ax = fig.add_subplot(111, aspect='equal')
        self._plot(ax, **kwargs)
        ax.margins(0.025, 0.025)
        self._makelegend(ax, **kwargs)
        if kwargs.get('scalebar', False):
            sb_kwg = dict(size=1, label='1mm', loc='lower right', frameon=False, color='k', label_top=True)
            sb_kwg.update(kwargs.get('scalebar_kwg', {}))
            sb_size = sb_kwg.pop('size')
            sb_label = sb_kwg.pop('label')
            sb_loc = sb_kwg.pop('loc')
            scalebar = AnchoredSizeBar(ax.transData, sb_size, sb_label, sb_loc, **sb_kwg)
            # scalebar = AnchoredSizeBar(ax.transData, 1, '1 mm', 'lower right', frameon=False, color='k', label_top=True)
            ax.add_artist(scalebar)
            ax.set_axis_off()
        else:
            ax.get_yaxis().set_tick_params(which='both', direction='out')
            ax.get_xaxis().set_tick_params(which='both', direction='out')
            plt.setp(ax.get_yticklabels(), rotation=90)
        return ax

    def show(self, **kwargs):
        """Show of ``Grains`` or ``Boundaries`` objects.

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
        # if 'ax' in kwargs:
        #     ax = kwargs.pop('ax')
        # fig = plt.figure()
        # ax = fig.add_subplot(111, aspect='equal')
        # self._plot(ax, **kwargs)
        # ax.margins(0.025, 0.025)
        # ax.get_yaxis().set_tick_params(which='both', direction='out')
        # ax.get_xaxis().set_tick_params(which='both', direction='out')
        # plt.setp(ax.get_yticklabels(), rotation=90)
        # self._makelegend(ax, **kwargs)
        self.plot(**kwargs)
        plt.savefig(kwargs.get('filename', 'figure.png'),
                    dpi=kwargs.get('dpi', 150))
        plt.close()

    def rose(self, **kwargs):
        """Plot polar histogram of ``Grains`` or ``Boundaries`` orientations

        Keywords:
          show: If True matplotlib show is called. Default True
          attr: property used for orientation. Default 'lao'
          bins: number of bins
          weights: if provided histogram is weighted
          density: True for probability density otherwise counts
          grid: True to show grid
          color: Bars color. Default is taken classification.
          ec: edgecolor. Default '#222222'
          alpha: alpha value. Default 1

        When show=False, returns matplotlib axes object

            """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            kwargs['legend'] = False  # figure not available
            kwargs['show'] = False    # likely another axes will be used before show
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)
        attr = kwargs.get('attr', 'lao')
        bins = kwargs.get('bins', 36)
        weights = kwargs.get('weights', [])
        grid = kwargs.get('grid', True)
        gridstep = kwargs.get('gridstep', 10)
        width = 360 / bins
        bin_edges = np.linspace(-width / 2, 360 + width / 2, bins + 2)
        bin_centres = (bin_edges[:-1] + np.diff(bin_edges) / 2)[:-1]
        bt = np.zeros(bins)
        for ix, key in enumerate(self.class_names):
            gix = self.classes(ix)
            ang = getattr(self[gix], attr)
            if 'weights' in kwargs:
                n, bin_edges = np.histogram(np.concatenate((ang, ang + 180)), bin_edges,
                                            weights=np.concatenate((weights[gix], weights[gix])),
                                            density=kwargs.get('density', False))
            else:
                n, bin_edges = np.histogram(np.concatenate((ang, ang + 180)), bin_edges,
                                            density=kwargs.get('density', False))
            # wrap
            n[0] += n[-1]
            n = n[:-1]
            if kwargs.get('scaled', True):
                n = np.sqrt(n)
            ax.bar(np.deg2rad(bin_centres), n,
                   width=np.deg2rad(width), bottom=bt,
                   color=kwargs.get('color', self.classes.color(key)),
                   label='{} ({})'.format(key, len(gix)),
                   edgecolor=kwargs.get('ec', '#222222'),
                   alpha=kwargs.get('alpha', 1))
            bt += n

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, gridstep), labels=np.arange(0, 360, gridstep))
        ax.set_rlabel_position(0)
        ax.grid(grid)
        if not grid:
            ax.get_yaxis().set_ticks([])
        if kwargs.get('legend', True):
            nr = np.ceil(len(self.class_names) / 3)
            fig.subplots_adjust(top=0.9 - 0.05 * nr)
            ax.legend(loc=9, borderaxespad=0., ncol=3, bbox_to_anchor=[0.5, 1.1 + 0.08 * nr])
        # plt.tight_layout()
        ax.set_axisbelow(True)
        if kwargs.get('show', True):
            plt.show()
        else:
            return ax

    def _seaborn_plot(self, sns_plot_fun, val, **kwargs):
        """Plot seaborn categorical plots.

        Keywords:
          show: If True matplotlib show is called. Default True
          attr: property used for plotting.
          hue: When True attr is used for hue and names for x.

        When show=False, returns matplotlib axes object.

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            kwargs['show'] = False    # likely another axes will be used before show
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        hue = kwargs.get('hue', False)
        if hue:
            sns_plot_fun(x='name', y=val,
                         data=self.df('class', 'name', val),
                         hue='class', hue_order=self.classes.index,
                         order=self.names,
                         palette=self.classes._colors_dict)
        else:
            sns_plot_fun(x='class', y=val,
                         data=self.df('class', val),
                         order=self.classes.index,
                         palette=self.classes._colors_dict)
        if kwargs.get('show', True):
            plt.show()
        else:
            return ax

    def barplot(self, val, **kwargs):
        """Plot seaborn swarmplot.

        """
        self._seaborn_plot(sns.barplot, val, **kwargs)

    def swarmplot(self, val, **kwargs):
        """Plot seaborn swarmplot.

        """
        self._seaborn_plot(sns.swarmplot, val, **kwargs)

    def boxplot(self, val, **kwargs):
        """Plot seaborn boxplot.

        """
        self._seaborn_plot(sns.boxplot, val, **kwargs)

    def violinplot(self, val, **kwargs):
        """Plot seaborn boxplot.

        """
        self._seaborn_plot(sns.violinplot, val, **kwargs)

    def countplot(self, **kwargs):
        """Plot seaborn countplot.

        """
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        hue = kwargs.get('hue', False)
        if hue:
            sns.countplot(x='name',
                          data=self.df('class', 'name'),
                          hue='class', hue_order=self.classes.index,
                          order=self.names,
                          palette=self.classes._colors_dict)
        else:
            sns.countplot(x='class',
                          data=self.df('class'),
                          order=self.classes.index,
                          palette=self.classes._colors_dict)
        if kwargs.get('show', True):
            plt.show()
        else:
            return ax

    def smooth(self, method='chaikin', **kwargs):
        return type(self)([getattr(s, method)(**kwargs) for s in self])

    def simplify(self, method='vw', **kwargs):
        return type(self)([getattr(s, method)(**kwargs) for s in self])

    def regularize(self, **kwargs):
        return type(self)([s.regularize(**kwargs) for s in self])


class Grains(PolySet):
    """Class to store set of ``Grains`` objects

    """
    def __repr__(self):
        # return 'Set of %s grains.' % len(self.polys)
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
    def circularity(self):
        """Return array of circularities (also called compactness) of the objects.
        circ = length**2/area

        """
        return np.array([p.circularity for p in self])

    @property
    def haralick(self):
        """Return array of Haralick’s circularities of the objects.
        hcirc = mean(R) / std(R) where R is array of centroid-vertex distances

        """
        return np.array([p.haralick for p in self])

    @property
    def aw_ead(self):
        """Returns normal area weighted mean of ead

        """
        loc, _ = weighted_avg_and_std(self.ead, self.area)
        return loc

    @property
    def aw_ead_log(self):
        """Returns lognormal area weighted mean of ead

        """
        loc, _ = weighted_avg_and_std(np.log10(self.ead), self.area)
        return 10**loc

    @property
    def rms_ead(self):
        """Returns root mean square of ead

        """
        return np.sqrt(np.mean(self.ead**2))

    def shape_vector(self, **kwargs):
        """Returns array of shape (feature) vectors.

        Keywords:
          N: number of points to regularize shape. Default 128
             Routine return N/2 of FDs

        """
        return np.array([p.shape_vector(**kwargs) for p in self])

    @property
    def nholes(self):
        """Returns array of number of holes (shape interiors)

        """
        return np.array([p.nholes for p in self])

    def boundaries_fast(self, T=None):
        """Create Boundaries from Grains. Faster but not always safe implementation

        Example:
          >>> g = Grains.from_shp()
          >>> b = g.boundaries_fast()

        """
        from shapely.ops import linemerge

        shapes = []
        lookup = {}
        if T is None:
            T = nx.Graph()
        G = nx.DiGraph()
        for fid, g in enumerate(self):
            # get name and add to list and legend
            path = []
            for co in g.shape.exterior.coords:
                if co not in lookup:
                    lookup[co] = len(lookup)
                path.append(lookup[co])
            G.add_path(path, fid=fid, name=g.name)
            for holes in g.shape.interiors:
                path = []
                for co in holes.coords:
                    if co not in lookup:
                        lookup[co] = len(lookup)
                    path.append(lookup[co])
                G.add_path(path, fid=fid, name=g.name)
        # Create topology graph
        H = G.to_undirected(reciprocal=True)
        # for edge in H.edges_iter():
        for edge in H.edges():
            e1 = G.get_edge_data(edge[0], edge[1])
            e2 = G.get_edge_data(edge[1], edge[0])
            bt = '%s-%s' % tuple(sorted([e1['name'], e2['name']]))
            T.add_node(e1['fid'])
            T.add_node(e2['fid'])
            T.add_edge(e1['fid'], e2['fid'], type=bt, bids=[])
        # Create boundaries
        # for edge in T.edges_iter():
        for edge in T.edges():
            shared = self[edge[0]].intersection(self[edge[1]])
            bt = T[edge[0]][edge[1]]['type']
            bid = len(shapes)
            if shared.geom_type == 'LineString':  # LineString cannot be merged
                shapes.append(Boundary(shared, bt, bid))
                T[edge[0]][edge[1]]['bids'] = [bid]
            else:
                # Skip if shared geometry is not line
                if shared.geom_type in ['LineString', 'MultiLineString', 'GeometryCollection']:
                    if 'Polygon' in [seg.geom_type for seg in shared]:
                        print('Overlap between polygons {} {}.'.format(edge[0], edge[1]))
                    # Skip points if polygon just touch
                    shared = linemerge([seg for seg in list(shared) if seg.geom_type == 'LineString'])
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
                        print('Wrong topology between polygons {} {}. Shared geometry is {}.'.format(edge[0], edge[1], shared.geom_type))
                else:
                    print('Wrong topology between polygons {} {}. Shared geometry is {}.'.format(edge[0], edge[1], shared.geom_type))
        if not shapes:
            print('No shared boundaries found.')
        else:
            return Boundaries(shapes)

    def boundaries(self, T=None):
        """Create Boundaries from Grains.

        Example:
          >>> g = Grains.from_shp()
          >>> b = g.boundaries()

        """
        from shapely.ops import linemerge

        def add_shared(gid, grain, oid, other, boundaries, T):
            shared = grain.intersection(other)
            if shared.geom_type in ['MultiLineString', 'GeometryCollection']:
                shared = linemerge([part for part in list(shared) if part.geom_type == 'LineString'])
            if shared.geom_type in ['MultiLineString', 'LineString']:
                if shared.geom_type == 'MultiLineString':
                    shared = list(shared)
                else:
                    shared = [shared]
                bids = []
                bt = '{}-{}'.format(*sorted([grain.name, other.name]))
                for bnd in shared:
                    bid = len(boundaries)
                    boundaries.append(Boundary(bnd, bt, bid))
                    bids.append(bid)
                T.add_node(oid, name=other.name)
                T.add_edge(gid, oid, type=bt, bids=bids)
            else:
                print('Unpredicted intersection geometry {} for polygons {}-{}'.format(shared.geom_type, gid, oid))

        if T is None:
            T = nx.Graph()

        allgrains = [(gid, grain) for gid, grain in enumerate(self)]
        boundaries = []
        while allgrains:
            gid, grain = allgrains.pop(0)
            T.add_node(gid, name=grain.name)
            for oid, other in allgrains:
                rel = grain.relate(other)
                if rel != 'FF2FF1212':  # disconnected
                    if rel == 'FF2F11212':  # shared boundary
                        add_shared(gid, grain, oid, other, boundaries, T)
                    elif rel == 'FF2F112F2':  # grain-incl
                        add_shared(gid, grain, oid, other, boundaries, T)
                    elif rel == 'FF2F1F212':  # incl-grain
                        add_shared(gid, grain, oid, other, boundaries, T)
                    elif rel == 'FF2F01212':  # Silently skip shared point for polygons
                        pass
                    elif rel == '212111212':
                        print('Skipping overlapping polygons {}-{}'.format(gid, oid))
                    else:
                        print('Hoops!!! Polygons {}-{} have relation {}'.format(gid, oid, rel))

        if not boundaries:
            print('No shared boundaries found.')
        else:
            return Boundaries(boundaries)

    @classmethod
    def from_shp(cls, filename=os.path.join(respath, 'sg2.shp'),
                 namefield='phase', name='None'):
        """Create Grains from ESRI shapefile.

        Args:
          filename: filename of shapefile. Default sg2.shp from examples
          namefield: name of attribute in shapefile that
            holds names of grains or None. Default "phase".
          name: value used for grain name when namefield is None

        """
        sf = Reader(filename)
        if sf.shapeType == 5:
            fieldnames = [field[0].lower() for field in sf.fields[1:]]
            if namefield is not None:
                if namefield in fieldnames:
                    name_pos = fieldnames.index(namefield)
                else:
                    raise Exception("There is no field '{}'. Available fields are: {}".format(namefield, fieldnames))
            shapeRecs = sf.shapeRecords()
            # until pyshp 2 will be released
            sf.shp.close()
            sf.shx.close()
            sf.dbf.close()
            shapes = []
            for pos, rec in enumerate(shapeRecs):
                # A valid polygon must have at least 4 coordinate tuples
                if len(rec.shape.points) > 3:
                    geom = shape(rec.shape.__geo_interface__)
                    # remove duplicate and subsequent colinear vertexes
                    # geom = geom.simplify(0)
                    # try  to "clean" self-touching or self-crossing polygons
                    if not geom.is_valid:
                        print('Cleaning FID={}...'.format(pos))
                        geom = geom.buffer(0)
                    if geom.is_valid:
                        if not geom.is_empty:
                            if namefield is None:
                                ph = name
                            else:
                                ph = rec.record[name_pos]
                            if geom.geom_type == 'MultiPolygon':
                                for g in geom:
                                    go = orient(g)
                                    if not any(go.equals(gr.shape) for gr in shapes):
                                        shapes.append(Grain(go, ph, len(shapes)))
                                    else:
                                        print('Duplicate polygon (FID={}) skipped.'.format(pos))
                                print('Multipolygon (FID={}) exploded.'.format(pos))
                            elif geom.geom_type == 'Polygon':
                                go = orient(geom)
                                if not any(go.equals(gr.shape) for gr in shapes):
                                    shapes.append(Grain(go, ph, len(shapes)))
                                else:
                                    print('Duplicate polygon (FID={}) skipped.'.format(pos))
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
            # until pyshp 2 will be released
            sf.shp.close()
            sf.shx.close()
            sf.dbf.close()
            raise Exception('Shapefile must contains polygons!')

    @classmethod
    def from_file(cls, filename=os.path.join(respath, 'sg2.shp'), **kwargs):
        """Create Grains from geospatial file.

        Args:
          filename: filename of shapefile. Default sg2.shp from examples
          namefield: name of attribute in shapefile that
            holds names of grains or None. Default "phase".
          name: value used for grain name when namefield is None

        """
        if fiona_OK:

            namefield = kwargs.pop('namefield', 'PHASE')
            name = kwargs.pop('name', 'None')

            with fiona.open(filename, **kwargs) as src:
                schema = src.schema
                assert schema['geometry'] == 'Polygon', 'The file must contains polygons!'
                fieldnames = list(schema['properties'].keys())
                if namefield is not None:
                    if namefield not in fieldnames:
                        raise Exception("There is no field '{}'. Available fields are: {}".format(namefield, fieldnames))
                shapes = []
                for feature in src:
                    geom = shape(feature['geometry'])
                    # remove duplicate and subsequent colinear vertexes
                    # geom = geom.simplify(0)
                    # try  to "clean" self-touching or self-crossing polygons
                    if not geom.is_valid:
                        print('Cleaning FID={}...'.format(feature['id']))
                        geom = geom.buffer(0)
                    if geom.is_valid:
                        if not geom.is_empty:
                            if namefield is None:
                                ph = name
                            else:
                                ph = feature['properties'][namefield]
                            if geom.geom_type == 'MultiPolygon':
                                for g in geom:
                                    go = orient(g)
                                    if not any(go.equals(gr.shape) for gr in shapes):
                                        shapes.append(Grain(go, ph, len(shapes)))
                                    else:
                                        print('Duplicate polygon (FID={}) skipped.'.format(feature['id']))
                                print('Multipolygon (FID={}) exploded.'.format(feature['id']))
                            elif geom.geom_type == 'Polygon':
                                go = orient(geom)
                                if not any(go.equals(gr.shape) for gr in shapes):
                                    shapes.append(Grain(go, ph, len(shapes)))
                                else:
                                    print('Duplicate polygon (FID={}) skipped.'.format(feature['id']))
                            else:
                                raise Exception('Unexpected geometry type (FID={})!'.format(feature['id']))
                        else:
                            print('Empty geometry (FID={}) skipped.'.format(feature['id']))
                    else:
                        print('Invalid geometry (FID={}) skipped.'.format(feature['id']))
                return cls(shapes)
        else:
            print('Fiona package is not installed.')

    def to_file(self, filename='grains.gpkg', driver='GPKG'):
        """
        driver: 'ESRI Shapefile', 'GeoJSON', 'GPKG' or 'GML'. Default 'GPKG'

        """
        if fiona_OK:
            _schema = {'geometry': 'Polygon', 'properties': OrderedDict([('id', 'int'), ('name', 'str')])}
            with fiona.open(filename, 'w', layer='grains', driver=driver, schema=_schema, crs={}) as dst:
                dst.writerecords(self.features)
        else:
            print('Fiona package is not installed.')

    def _plot(self, ax, **kwargs):
        alpha = kwargs.get('alpha', 0.8)
        ec = kwargs.get('ec', '#222222')
        legend = kwargs.get('legend', True)
        groups = self.groups('shape')
        keys = groups.groups.keys()
        for key in self.class_names:
            paths = []
            if key in keys:
                group = groups.get_group(key)
                for g in group['shape']:
                    paths.append(PolygonPath(g))
                if legend:
                    patch = PathPatch(Path.make_compound_path(*paths),
                                      fc=self.classes.color(key),
                                      ec=ec, alpha=alpha, zorder=2,
                                      label='{} ({})'.format(key, len(group)))
                else:
                    patch = PathPatch(Path.make_compound_path(*paths),
                                      fc=self.classes.color(key),
                                      ec=ec, alpha=alpha, zorder=2)
            else:
                if legend:
                    patch = PathPatch(Path([[None, None]]),
                                      fc=self.classes.color(key),
                                      ec=ec, alpha=alpha, zorder=2,
                                      label='{} ({})'.format(key, 0))
            ax.add_patch(patch)
        if kwargs.get('show_index', False):
            for idx, p in enumerate(self):
                ax.text(p.xc, p.yc, str(idx),
                        bbox=dict(facecolor='yellow', alpha=0.5))
        if kwargs.get('show_fid', False):
            for p in self:
                ax.text(p.xc, p.yc, str(p.fid),
                        bbox=dict(facecolor='yellow', alpha=0.5))
        return ax

    def grainsize_plot(self, areaweighted=True, **kwargs):
        from .plots import grainsize_plot
        if 'weights' in kwargs:
            _ = kwargs.pop('weights')
        if 'title' not in kwargs:
            kwargs['title'] = 'Grainsize plot [EAD]'
        if areaweighted:
            grainsize_plot(self.ead, weights=self.area, **kwargs)
        else:
            grainsize_plot(self.ead, **kwargs)

    def areafraction_plot(self, **kwargs):
        if 'title' not in kwargs:
            kwargs['title'] = 'Area fraction plot'
        bins = kwargs.get('bins', 'auto')
        title = kwargs.get('title', None)
        xlog = kwargs.get('log', False)
        d = self.ead
        ld = np.log10(d)
        areas = self.area
        # rms = np.sqrt(np.mean(d**2))
        if 'ax' in kwargs:
            ax = kwargs.pop('ax')
            show = False
        else:
            f, ax = plt.subplots(figsize=kwargs.get('figsize', plt.rcParams.get('figure.figsize')))
            show = True
        if xlog:
            bin_edges = np.histogram_bin_edges(ld, bins='auto')
            inds = np.digitize(ld, bin_edges, right=False)
            # statistics
            loc, scale = weighted_avg_and_std(ld, areas)
            # default left right values
            left = kwargs.get('left', 10**(loc - 3.5*scale))
            right = kwargs.get('right', 10**(loc + 3.5*scale))
        else:
            bin_edges = np.histogram_bin_edges(d, bins=bins)
            inds = np.digitize(d, bin_edges, right=False)
            bw = bin_edges[1:] - bin_edges[:-1]
            left = kwargs.get('left', bin_edges[0] - bw[0])
            right = kwargs.get('right', bin_edges[-1] + bw[-1])
        # include right to last bin
        inds[inds == len(bin_edges)] = len(bin_edges) - 1
        if xlog:
            bin_edges = 10**bin_edges
        bw = bin_edges[1:] - bin_edges[:-1]
        bc = (bin_edges[:-1] + bin_edges[1:]) / 2

        # area fractions
        af = []
        for ind in range(1, len(bin_edges)):
            if ind in inds:
                af.append(areas[inds == ind].sum())
            else:
                af.append(0)
        af = 100 * np.array(af) / sum(areas)
        # plot
        ax.bar(bc, af, width=0.9*bw, color='mediumseagreen')
        if xlog:
            ax.set_xscale('log')
        ax.set_xlim(left=left, right=right)
        if show:
            if title is not None and show:
                f.suptitle(title)
            ax.set_xlabel('EAD')
            ax.set_ylabel('Area fraction [%]')
            plt.show()


class Boundaries(PolySet):
    """Class to store set of ``Boundaries`` objects

    """
    def __repr__(self):
        # return 'Set of %s boundaries.' % len(self.polys)
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

    @classmethod
    def from_shp(cls, filename=None, namefield='phase', name='None'):
        """Create Boundaries from ESRI shapefile.

        Args:
          filename: filename of shapefile.
          namefield: name of attribute in shapefile that
            holds names of boundairies or None. Default "phase".
          name: value used for grain name when namefield is None

        """
        sf = Reader(filename)
        if sf.shapeType == 3:
            fieldnames = [field[0].lower() for field in sf.fields[1:]]
            if namefield is not None:
                if namefield in fieldnames:
                    name_pos = fieldnames.index(namefield)
                else:
                    raise Exception("There is no field '%s'. Available fields are: %s" % (namefield, fieldnames))
            shapeRecs = sf.shapeRecords()
            # until pyshp 2 will be released
            sf.shp.close()
            sf.shx.close()
            sf.dbf.close()
            shapes = []
            for pos, rec in enumerate(shapeRecs):
                # A valid polyline must have at least 2 coordinate tuples
                if len(rec.shape.points) > 1:
                    geom = shape(rec.shape.__geo_interface__)
                    if geom.is_valid:
                        if not geom.is_empty:
                            if namefield is None:
                                ph = name
                            else:
                                ph = rec.record[name_pos]
                            if geom.geom_type == 'MultiLineString':
                                for g in geom:
                                    shapes.append(Boundary(g, ph, len(shapes)))
                                print('Multiline (FID={}) exploded.'.format(pos))
                            elif geom.geom_type == 'LineString':
                                shapes.append(Boundary(geom, ph, len(shapes)))
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
            # until pyshp 2 will be released
            sf.shp.close()
            sf.shx.close()
            sf.dbf.close()
            raise Exception('Shapefile must contains polylines!')

    def to_file(self, filename='boundaries.gpkg', driver='GPKG'):
        """
        driver: 'ESRI Shapefile', 'GeoJSON', 'GPKG' or 'GML'. Default 'GPKG'

        """
        import fiona
        _schema = {'geometry': 'LineString', 'properties': OrderedDict([('id', 'int'), ('name', 'str')])}
        with fiona.open(filename, 'w', layer='boundaries', driver=driver, schema=_schema, crs={}) as dst:
            dst.writerecords(self.features)

    def _plot(self, ax, **kwargs):
        alpha = kwargs.get('alpha', 0.8)
        legend = kwargs.get('legend', True)
        groups = self.groups('shape')
        for key in self.class_names:
            group = groups.get_group(key)
            x = []
            y = []
            for b in group['shape']:
                xb, yb = b.xy
                x.extend(xb)
                x.append(np.nan)
                y.extend(yb)
                y.append(np.nan)
            if legend:
                ax.plot(x, y, color=self.classes.color(key), alpha=alpha,
                        label='{} ({})'.format(key, len(group)))
            else:
                ax.plot(x, y, color=self.classes.color(key), alpha=alpha)
        if kwargs.get('show_index', False):
            for idx, p in enumerate(self):
                ax.text(p.xc, p.yc, str(idx),
                        bbox=dict(facecolor='white', alpha=0.5))
        if kwargs.get('show_fid', False):
            for p in self:
                ax.text(p.xc, p.yc, str(p.fid),
                        bbox=dict(facecolor='white', alpha=0.5))
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
                 namefield='phase', name=''):
        return cls.from_grains(Grains.from_shp(filename, namefield), name=name)

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
                tri.update({G[n0][n1]['bid']})
            res.append(list(tri))
        return res

    def bids(self, idx, name=None):
        """Return array of indexes of boundaries creating grain idx

        If name keyword is provided only boundaries with grains of
        given name are returned.

        """
        nids = self.neighbors(idx, name=name)
        bids = []
        for nid in nids:
            bids.extend(self.T[idx][nid]['bids'])
        return bids

    def get_cluster(self, idx, name=None):
        """Return array of indexes of clustered grains seeded from idx.

        If name keyword is provided only neighbours with given name
        are returned.

        """
        last = 0
        cluster = set([idx])
        while len(cluster) > last:
            last = len(cluster)
            for idx in list(cluster):
                cluster.update(self.neighbors(idx, name=name))
        return list(cluster)

    def get_clusters(self):
        """Return dictionary with lists of clusters for each name.

        """
        res = {}
        for name in self.g.names:
            aid = set(self.g.getindex(name))
            clusters = []
            while aid:
                cluster = self.get_cluster(aid.pop(), name=name)
                aid = aid.difference(cluster)
                clusters.append(list(cluster))
            res[name] = clusters
        return res

    def dissolve(self):
        grains = []
        fid = 0
        clusters = self.get_clusters()
        for name in clusters:
            for cidx in clusters[name]:
                grains.append(Grain(cascaded_union([g.shape for g in self.g[cidx]]), name, fid))
                fid += 1
        return Grains(grains)

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
          alpha: Grains transparency. Default 0.8
          pos: legend position "top" or "right". Defalt Auto
          ncol: number of columns for legend.
          show_fid: Show FID of objects. Default False
          show_index: Show index of objects. Default False

        Returns matplotlib axes object.

        """
        if 'ax' in kwargs:
            ax = kwargs['ax']
            ax.set_aspect('equal')
        else:
            fig = plt.figure(figsize=kwargs.get('figsize', plt.rcParams.get('figure.figsize')))
            ax = fig.add_subplot(111, aspect='equal')
        self.g._plot(ax, **kwargs)
        # non transparent bounbdaries
        kwargs['alpha'] = 1
        self.b._plot(ax, **kwargs)
        plt.setp(ax.get_yticklabels(), rotation=90)
        self.g._makelegend(ax, **kwargs)
        ax.set_ylabel(self.name)
        return ax

    def show(self, **kwargs):
        """Show plot of ``Sample`` objects.

        """
        self.plot(**kwargs)
        plt.show()
