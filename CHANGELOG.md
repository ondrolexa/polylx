# Changes

### 0.6.1 (24 Jan 2025)
 * csd_plot() is interactive only from terminal
 * contact_frequency() bug fix

### 0.6.0 (22 Jan 2025)
 * fiona added to dependencies
 * csd_plot after Peterson 1996 added
 * contact_frequency Grains method added
 * utils.circstat provides most common circular statistics
   suitable for pandas agg

### 0.5.5 (14 Dec 2024)
 * from_coords method added to Boundary
 * added eap and epa Grain methods
 * surfor for Grains normalized by factor 2
 * vertex_angles property added
 * ortensor added to utils
 * agg accepts kwargs allowing define names of aggregated columns

### 0.5.4 (05 Mar 2024)
 * shapelysmooth methods added for smoothing
 * shapely and scipy upstream fixes
 * jenks and quantile rules fix
 * bcov shape_method added for eigenanalysis of decomposed geometry

### 0.5.3 (06 Mar 2023)
 * upstream fix for networkX 3
 * Fracnet.from_boundaries bug fixed

### 0.5.2 (06 Mar 2023)

 * upstream fix for shapely 3
 * topological analyses added to Fracnet

### 0.5.1 (27 May 2021)

 * fourier_ellipse shape method for Grains added
 * eliptic fourier smoothing for Grains added
 * added grainsize plot
 * added accumulate method to Grains and Boundaries
 * simple fiona reader implemented (fiona must be installed)
 * added kde plot

## 0.5 (29 Jan 2019)

 * rose plot groupped according to classification
 * get_class, class_iter methods added to Grains and Boundaries
 * seaborn added to requirements
 * several seaborn categorical plots are added as methods
   (swarmplot, boxplot, barplot, countplot)

### 0.4.9 (12 Dec 2017)

* getindex method of Grains and Boundaries implemented
* Grain cdist property return centroid-vertex distance function
* Grain cdir property return centroid-vertex direction function
* Grain shape_vector property returns normalized Fourier descriptors
* Grain regularize method returns Grain with regularly distributed vertices
* Classification could be based on properties or any other values
* boundary_segments method added
* Smoothing, simplification and regularization of boundaries implemented
* Colortable for legend is persistant trough indexing. Classify method
  could be used to change it
* Default color table is seaborn muted for unique classification
  and matplotlib viridis for continuous classes

### 0.4.8 (04 Mar 2017)

* bugfix

### 0.4.6 (04 Mar 2017)

* added plots module (initial)
* representative_point for Grains implemented
* moments calculation including holes
* surfor and parror functions added
* orientation of polygons is unified and checked
* minbox shape method added

### 0.4.5 (12 Jan 2017)

* shell script ipolylx opens interactive console

### 0.4.4 (12 Jan 2017)

* Added MAEE (minimum area enclosing ellipse) to grain shape methods
* Removed embedded IPython and IPython requirements

### 0.4.3 (02 Sep 2016)

* IPython added to requirements

### 0.4.2 (02 Sep 2016)

* Sample has pairs property(dictionary) to map boundary id to grains id
* Sample triplets method returns list of grains id creating triple points

### 0.4.1 (20 Jun 2016)

* Examples added to distribution

## 0.4 (20 Jun 2016)

* Sample neighbors_dist method to calculate neighbors distances
* Grains and Boundaries nndist to calculate nearest neighbors distances
* Fancy indexing with slices fixed
* Affine transformations affine_transform, rotate, scale, skew, translate
  methods implemented for Grains and Boundaries
* Sample name atribute added
* Sample bids method to get boundary id's related to grain added

### 0.3.2 (04 Jun 2016)

* PolyShape name forced to be string
* Creation of boundaries is Grains method

### 0.3.1 (22 Feb 2016)

* classification is persitant trough fancy indexing
* empty classes allowed
* bootstrap method added to PolySet

## 0.2 (18 Apr 2015)

* Smooth and simplify methods for Grains implemented
* Initial documentation added
* `phase` and `type` properties renamed to `name`

## 0.1 (13 Feb 2015)

* First release
