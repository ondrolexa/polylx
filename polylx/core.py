# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:42:54 2014

@author: ondro

Example:

from core import *
g = Grains.from_shp('m1-p')
b = Boundaries.from_grains(g)

from core import *
from shapely.geometry import Polygon
g = Grain('rect',Polygon([(2, 0), (0, 4), (8, 8), (10,4)]))

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import shape
import networkx as nx
import pandas as pd
import itertools
import os

from .shapefile import Reader
from .utils import natural_breaks, fisher_jenks
from .utils import PolygonPath

from pkg_resources import resource_filename

# lambda funkce
sind = lambda x: np.sin(np.deg2rad(x))
cosd = lambda x: np.cos(np.deg2rad(x))
tand = lambda x: np.tan(np.deg2rad(x))
asind = lambda x: np.rad2deg(np.arcsin(x))
acosd = lambda x: np.rad2deg(np.arccos(x))
atand = lambda x: np.rad2deg(np.arctan(x))
atan2d = lambda x1, x2: np.rad2deg(np.arctan2(x1, x2))


class PolyShape(object):

    def __init__(self):
        self.shape_method = 'maxferet'
        # x, y = self.xy
        # x = x - x.mean()
        # y = y - y.mean()
        # xl = np.roll(x, -1)
        # yl = np.roll(y, -1)
        # v = xl*y - x*yl
        # A = np.sum(v)/2
        # if A != 0:
        #     a10 = np.sum(v*(xl + x))/(6*A)
        #     a01 = np.sum(v*(yl + y))/(6*A)
        #     a20 = np.sum(v*(xl**2 + xl*x + x**2))/(12*A)
        #     a11 = np.sum(v*(2*xl*yl + xl*y + x*yl + 2*x*y))/(24*A)
        #     a02 = np.sum(v*(yl**2 + yl*y + y**2))/(12*A)

        #     m20 = a20 - a10**2
        #     m11 = a11 - a10*a01
        #     m02 = a02 - a01**2

        #     CM = np.array([[m02, -m11], [-m11, m20]])/(4*(m20*m02 - m11**2))
        #     evals, evecs = np.linalg.eig(CM)
        #     idx = evals.argsort()
        #     self.evals = evals[idx]
        #     self.evecs = evecs[:, idx].T
        # else:
        #     v1 = np.array([x[-1] - x[0], y[-1] - y[0]])
        #     v1 = v1/np.sqrt(v1.dot(v1))
        #     v2 = np.roll(v1, 1)
        #     v2[0] = -v2[0]
        #     CM = np.zeros((2, 2))
        #     self.evals = np.zeros(2)
        #     self.evecs = np.array([v1, v2])

    def __getattr__(self, attr):
        if hasattr(self.shape, attr):
            return getattr(self.shape, attr)
        else:
            raise AttributeError

    @property
    def shape_method(self):
        return self._shape_method

    @shape_method.setter
    def shape_method(self, value):
        getattr(self, value)()   # to evaluate and check validity
        self._shape_method = value

    @property
    def ar(self):
        return self.la/self.sa

    @property
    def ma(self):
        return np.sqrt(self.la*self.sa)

    @property
    def centroid(self):
        return self.shape.centroid.coords[0]

    def feret(self, angle=0):
        pp = np.dot(self.hull.T, np.array([sind(angle), cosd(angle)]))
        return pp.max(axis=0) - pp.min(axis=0)

    ###############################################################
    #       Shape methods (should modify sa, la, sao, lao, xc, yc)
    ###############################################################
    def minferet(self):
        # return tuple of minimum feret and orientation
        xy = self.hull.T
        dxy = xy[1:]-xy[:-1]
        ang = (atan2d(*dxy.T) + 90) % 180
        pp = np.dot(xy, np.array([sind(ang), cosd(ang)]))
        d = pp.max(axis=0) - pp.min(axis=0)
        self.sa = np.min(d)
        self.sao = ang[d.argmin()]
        self.lao = (self.sao + 90) % 180
        self.la = self.feret(self.lao)
        self.xc, self.yc = self.shape.centroid.coords[0]

    def maxferet(self):
        # return tuple of maximum feret and orientation
        xy = self.hull.T
        pa = np.array(list(itertools.combinations(range(len(xy)), 2)))
        d2 = np.sum((xy[pa[:, 0]] - xy[pa[:, 1]])**2, axis=1)
        ix = d2.argmax()
        dxy = xy[pa[ix][1]]-xy[pa[ix][0]]
        self.la = np.sqrt(np.max(d2))
        self.lao = atan2d(*dxy) % 180
        self.sao = (self.lao + 90) % 180
        self.sa = self.feret(self.sao)
        self.xc, self.yc = self.shape.centroid.coords[0]
    ###############################################################


class Grain(PolyShape):

    def __init__(self, phase, shape):
        self.shape = shape
        self.phase = phase
        super(Grain, self).__init__()

    def __repr__(self):
        return 'Grain [%s] la:%g, sa:%g, lao:%g, sao:%g (%s)' % \
            (self.phase, self.la, self.sa, self.lao, self.sao, self.shape_method)

    @property
    def xy(self):
        return np.array(self.shape.exterior.xy)

    @property
    def hull(self):
        return np.array(self.shape.convex_hull.exterior.xy)

    @property
    def area(self):
        return self.shape.area

    @property
    def perimeter(self):
        return self.shape.length

    @property
    def ead(self):
        return 2*np.sqrt(self.shape.area/np.pi)

    def show(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        hull = self.hull
        ax.plot(*hull, ls='--', c='g')
        ax.add_patch(PathPatch(PolygonPath(self.shape),
                     fc='blue', ec='#000000', alpha=0.5, zorder=2))
        pa = np.array(list(itertools.combinations(range(len(hull.T)), 2)))
        d2 = np.sum((hull.T[pa[:, 0]] - hull.T[pa[:, 1]])**2, axis=1)
        ix = d2.argmax()
        ax.plot(*hull.T[pa[ix]].T, ls=':', lw=2, c='r')
        ax.autoscale_view(None, True, True)
        plt.show()


class Boundary(PolyShape):

    def __init__(self, typ, shape):
        self.shape = shape
        self.type = typ
        super(Boundary, self).__init__()

    def __repr__(self):
        return 'Boundary [%s] la:%g, sa:%g, lao:%g, sao:%g (%s)' % \
            (self.type, self.la, self.sa, self.lao, self.sao, self.shape_method)

    @property
    def xy(self):
        return np.array(self.shape.xy)

    @property
    def hull(self):
        h = self.shape.convex_hull
        if h.geom_type == 'LineString':
            return np.array(h.xy)[:, [0, 1, 0]]
        else:
            return np.array(h.exterior.xy)

    @property
    def length(self):
        return self.shape.length

    def show(self):
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
        plt.show()


class PolySet(object):

    def __init__(self, shapes):
        self.polys = shapes
        self.shape_method = 'maxferet'
        gb = self.bounds
        self.extent = gb[:, 0].min(), gb[:, 1].min(), gb[:, 2].max(), gb[:, 3].max()

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
            res = np.array([getattr(p, attr) for p in self])
        return res

    @property
    def shape_method(self):
        return self._shape_method

    @shape_method.setter
    def shape_method(self, value):
        for p in self:
            if not hasattr(p, '_shape_method'):
                p.shape_method = value
        self._shape_method = value

    @property
    def width(self):
        return self.extent[2] - self.extent[0]

    @property
    def height(self):
        return self.extent[3] - self.extent[1]

    def feret(self, angle=0):
        return np.array([p.feret(angle) for p in self])

    def classify(self, attr, rule='natural', k=5):
        self.class_attr = attr
        #vals = [getattr(p, attr) for p in self]
        vals = getattr(self, attr)
        if rule == 'unique':
            self.classes = vals
            self.class_index, index = np.unique(vals, return_inverse=True)
            counts = np.bincount(index)
            self.class_legend = ['%s (%d)' % p for p in zip(self.class_index, counts)]
        elif rule == 'equal':
            counts, bins = np.histogram(vals, k)
            index = np.digitize(vals, bins) - 1
            index[np.flatnonzero(index == k)] = k - 1
            self.class_labels = ['%g-%g (%d)' % (bins[i], bins[i+1], count) for i, count in enumerate(counts)]
            self.class_index = ['%g-%g' % (bins[i], bins[i+1]) for i in range(len(counts))]
            self.classes = np.array([self.class_index[i] for i in index])
        elif rule == 'natural':
            index, bins = natural_breaks(vals, k=k)
            counts = np.bincount(index)
            self.class_labels = ['%g-%g (%d)' % (bins[i], bins[i+1], count) for i, count in enumerate(counts)]
            self.class_index = ['%g-%g' % (bins[i], bins[i+1]) for i in range(len(counts))]
            self.classes = np.array([self.class_index[i] for i in index])
        elif rule == 'jenks':
            bins = fisher_jenks(vals, k=k)
            index = np.digitize(vals, bins) - 1
            index[np.flatnonzero(index == k)] = k - 1
            counts = np.bincount(index)
            self.class_labels = ['%g-%g (%d)' % (bins[i], bins[i+1], count) for i, count in enumerate(counts)]
            self.class_index = ['%g-%g' % (bins[i], bins[i+1]) for i in range(len(counts))]
            self.classes = np.array([self.class_index[i] for i in index])

    def df(self, *attrs):
        d = pd.DataFrame()
        for attr in attrs:
            #d[attr] = [getattr(p, attr) for p in self]
            d[attr] = getattr(self, attr)
        return d

    def agg(self, aggfunc, *attrs):
        return getattr(self.groups(*attrs), aggfunc)().reindex(self.class_index)

    def groups(self, *attrs):
        df = self.df(*attrs)
        return df.groupby(self.classes)

    def df_c(self, *attrs):
        d = pd.DataFrame({self.class_attr + '_class': self.classes})
        for attr in attrs:
            d[attr] = [getattr(p, attr) for p in self]
        return d

    def _autocolortable(self, cmap='jet'):
        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
        n = len(self.class_index)
        if n > 1:
            pos = np.round(np.linspace(0, cmap.N - 1, n))
        else:
            pos = [127]
        return dict(zip(self.class_index, [cmap(int(i)) for i in pos]))

    def _makelegend(self, ax, loc='auto', ncol=3):
        if loc == 'auto':
            if self.width > self.height:
                loc = 'top'
                ncol = 3
            else:
                loc = 'right'
                ncol = 1
        if loc == 'top':
            h, l = ax.get_legend_handles_labels()
            lgd = plt.figlegend(h, l, loc='upper center', bbox_to_anchor=[0.5, 0.99], ncol=ncol)
            plt.draw()
            prop = lgd.get_window_extent().height/ax.get_figure().get_window_extent().height
            ax.get_figure().tight_layout(rect=[0.02, 0.02, 0.98, 0.98 - prop])
        elif loc == 'right':
            h, l = ax.get_legend_handles_labels()
            lgd = plt.figlegend(h, l, loc='center right', bbox_to_anchor=[0.99, 0.5], ncol=ncol)
            plt.draw()
            prop = lgd.get_window_extent().width/ax.get_figure().get_window_extent().width
            ax.get_figure().tight_layout(rect=[0.02, 0.02, 0.98 - prop, 0.98])
        else:
            lgd = None
        return lgd

    def plot(self, legend=None, loc='auto', alpha=0.8, cmap='jet', ncol=1):
        if legend is None:
            legend = self._autocolortable(cmap)
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        self._plot(ax, legend, alpha)
        plt.setp(plt.yticks()[1], rotation=90)
        self._makelegend(ax, loc, ncol)

    def savefig(self, filename='grains.png', legend=None, loc='auto', alpha=0.8, cmap='jet', dpi=150, ncol=1):
        if legend is None:
            legend = self._autocolortable(cmap)
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        self._plot(ax, legend, alpha)
        plt.setp(plt.yticks()[1], rotation=90)
        self._makelegend(ax, loc, ncol)
        plt.savefig(filename, dpi=dpi)
        plt.close()

    def rose(self, ang=None, bins=36, scaled=True, weights=None, density=False, arrow=0.95, rwidth=1, **kwargs):
        if ang is None:
            ang = self.lao
        #plot
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        width = 360/bins
        num, bin_edges = np.histogram(np.concatenate((ang, ang + 180)), bins=bins + 1, range=(-width/2, 360 + width/2), weights=weights, density=density)
        num[0] += num[-1]
        num = num[:-1]
        if scaled:
            num = np.sqrt(num)
        theta = []
        radii = []
        for cc, val in zip(np.arange(0, 360, width), num):
            theta.extend([cc - width/2, cc - rwidth*width/2, cc, cc + rwidth*width/2, cc + width/2, ])
            radii.extend([0, val*arrow, val, val*arrow, 0])

        ax.fill(np.deg2rad(theta), radii, **kwargs)
        # IDEA

        #ax = plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)
        #ax.set_theta_zero_location('N')
        #ax.set_theta_direction(-1)
        #bars = plt.bar(theta, radii, width=width*0.8, alpha=0.5, align='center')
        #ax.set_yticklabels([])

        # IDEA PDF
        #x=np.linspace(-np.pi, np.pi, 1801)
        #ax = plt.subplot(111, polar=True)
        #kappa = 250
        #pdf = np.zeros_like(x)
        #for lao in g.lao:
        #    pdf += vonmises.pdf(x, kappa, loc=np.radians(lao))
        #    pdf += vonmises.pdf(x, kappa, loc=np.radians(lao+180))

        #pdf /= len(g.lao)
        #ax.plot(x+np.pi, np.sqrt(pdf), label='%s'%kappa)
        #ax.set_theta_zero_location('N')
        #ax.set_theta_direction(-1)
        #ax.grid(True)


class Grains(PolySet):

    def __init__(self, shapes):
        super(Grains, self).__init__(shapes)
        self.classify('phase', 'unique')

    def __repr__(self):
        return 'Set of %s grains.' % len(self.polys)

    def __add__(self, other):
        return Grains(self.polys + other.polys)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.polys[index]
        elif isinstance(index, str):
            return Grains([g for g in self.polys if g.phase == index])
        elif isinstance(index, np.ndarray):
            if index.dtype == 'bool':
                index = np.flatnonzero(index)
            return Grains([self.polys[fid] for fid in index])
        else:
            return Grains(self.polys[index])

    def phase_list(self, unique=True):
        pl = list(self.phase)
        if unique:
            return sorted(list(set(pl)))
        else:
            return pl

    def phase_dict(self):
        return {key: g.phase for (key, g) in enumerate(self)}

    @classmethod
    def from_shp(self, filename=os.path.join(resource_filename(__name__, 'example'), 'sg2.shp'), phasefield='phase'):
        sf = Reader(filename)
        if sf.shapeType == 5:
            fieldnames = [field[0].lower() for field in sf.fields[1:]]
            if phasefield in fieldnames:
                phase_pos = fieldnames.index(phasefield)
            else:
                raise Exception("There is no field '%s'. Available fields are: %s" % (phasefield, fieldnames))
            shapeRecs = sf.shapeRecords()
            shapes = []
            for pos, rec in enumerate(shapeRecs):
                geom = shape(rec.shape.__geo_interface__)
                # try  to "clean" self-touching or self-crossing polygons such as the classic "bowtie".
                if not geom.is_valid:
                    geom = geom.buffer(0)
                if geom.is_valid:
                    if not geom.is_empty:
                        if geom.geom_type == 'MultiPolygon':
                            for g in geom:
                                shapes.append(Grain(rec.record[phase_pos], g))
                            print('Multipolygon (FID={}) exploded.'.format(pos))
                        elif geom.geom_type == 'Polygon':
                            shapes.append(Grain(rec.record[phase_pos], geom))
                        else:
                            raise Exception('Unexpected geometry type (FID={})!'.format(pos))
                    else:
                        print('Empty geometry (FID={}) skipped.'.format(pos))
                else:
                    print('Invalid geometry (FID={}) skipped.'.format(pos))
            return self(shapes)
        else:
            raise Exception('Shapefile must contains polygons!')

    def _plot(self, ax, legend, alpha, ec='#222222'):
        groups = self.groups('shape')
        for key in self.class_index:
            group = groups.get_group(key)
            paths = []
            for g in group['shape']:
                paths.append(PolygonPath(g))
            patch = PathPatch(Path.make_compound_path(*paths), fc=legend[key], ec=ec, alpha=alpha, zorder=2, label='{} ({})'.format(key, len(group)))
            ax.add_patch(patch)
        ax.margins(0.025, 0.025)
        ax.get_yaxis().set_tick_params(which='both', direction='out')
        ax.get_xaxis().set_tick_params(which='both', direction='out')
        return ax


class Boundaries(PolySet):

    def __init__(self, shapes):
        super(Boundaries, self).__init__(shapes)
        self.classify('type', 'unique')

    def __repr__(self):
        return 'Set of %s boundaries.' % len(self.polys)

    def __add__(self, other):
        return Boundaries(self.polys + other.polys)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.polys[index]
        elif isinstance(index, str):
            okindex = '%s-%s' % tuple(sorted(index.split('-')))
            return Boundaries([b for b in self.polys if b.type == okindex])
        elif isinstance(index, np.ndarray):
            if index.dtype == 'bool':
                index = np.flatnonzero(index)
            return Boundaries([self.polys[fid] for fid in index])
        else:
            return Boundaries(self.polys[index])

    def type_list(self, unique=True):
        pl = list(self.type)
        if unique:
            return sorted(list(set(pl)))
        else:
            return pl

    def type_dict(self):
        return {key: b.type for (key, b) in enumerate(self)}

    @classmethod
    def from_grains(self, grains, T=None):
        from shapely.ops import linemerge

        shapes = []
        lookup = {}
        if T is None:
            T = nx.Graph()
        G = nx.DiGraph()
        for fid, g in enumerate(grains.polys):
            # get phase and add to list and legend
            path = []
            for co in g.shape.exterior.coords:
                if not co in lookup:
                    lookup[co] = len(lookup)
                path.append(lookup[co])
            G.add_path(path, fid=fid, phase=g.phase)
            for holes in g.shape.interiors:
                path = []
                for co in holes.coords:
                    if not co in lookup:
                        lookup[co] = len(lookup)
                    path.append(lookup[co])
                G.add_path(path, fid=fid, phase=g.phase)
        # Create topology graph
        H = G.to_undirected(reciprocal=True)
        for edge in H.edges_iter():
            e1 = G.get_edge_data(edge[0],edge[1])
            e2 = G.get_edge_data(edge[1],edge[0])
            bt = '%s-%s' % tuple(sorted([e1['phase'], e2['phase']]))
            T.add_node(e1['fid'])
            T.add_node(e2['fid'])
            T.add_edge(e1['fid'], e2['fid'], type=bt)
        # Create boundaries
        for edge in T.edges_iter():
            shared = grains[edge[0]].intersection(grains[edge[1]])
            typ = T[edge[0]][edge[1]]['type']
            if shared.geom_type == 'LineString':
                shapes.append(Boundary(typ, shared))
            elif shared.geom_type == 'MultiLineString':
                shared = linemerge(shared)
                if shared.geom_type == 'LineString':
                    shapes.append(Boundary(typ, shared))
                else:
                    for sub in list(shared):
                        shapes.append(Boundary(typ, sub))
            elif shared.geom_type == 'GeometryCollection':
                for sub in shared:
                    if sub.geom_type == 'LineString':
                        shapes.append(Boundary(typ, sub))
                    elif sub.geom_type == 'MultiLineString':
                        sub = linemerge(sub)
                        if sub.geom_type == 'LineString':
                            shapes.append(Boundary(typ, sub))
                        else:
                            for subsub in list(sub):
                                shapes.append(Boundary(typ, subsub))
        return self(shapes)

    def _plot(self, ax, legend, alpha):
        groups = self.groups('shape')
        for key in self.class_index:
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

    def __init__(self):
        self.g = None
        self.b = None
        self.T = None

    def __repr__(self):
        return 'Sample with %s grains and %s boundaries.' % (len(self.g.polys),len(self.b.polys))

    @classmethod
    def from_shp(cls, filename=os.path.join(resource_filename(__name__, 'example'), 'sg2.shp'), phasefield='phase'):
        return cls.from_grains(Grains.from_shp(filename, phasefield))

    @classmethod
    def from_grains(cls, grains):
        obj = cls()
        obj.T = nx.Graph()
        obj.g = grains
        obj.b = Boundaries.from_grains(grains, obj.T)
        return obj

    def plot(self, legend=None, loc='auto', alpha=0.8, cmap='jet', ncol=1):
        if legend is None:
            legend = dict(list(self.g._autocolortable().items()) + list(self.b._autocolortable().items()))
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        self.g._plot(ax, legend, alpha, ec='none')
        self.b._plot(ax, legend, 1)
        plt.setp(plt.yticks()[1], rotation=90)
        self.g._makelegend(ax, loc, ncol)
