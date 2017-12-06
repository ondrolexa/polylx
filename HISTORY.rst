.. :changelog:

Changes
=======

0.1 (13 Feb 2015)
-----------------
* First release

0.2 (18 Apr 2015)
-----------------
* Smooth and simplify methods for Grains implemented
* Initial documentation added
* `phase` and `type` properties renamed to `name`

0.3 (22 Feb 2016)
-----------------

0.3.1 (22 Feb 2016)
~~~~~~~~~~~~~~~~~~~
* classification is persitant trough fancy indexing
* empty classes allowed
* bootstrap method added to PolySet

0.3.2 (04 Jun 2016)
~~~~~~~~~~~~~~~~~~~
* PolyShape name forced to be string
* Creation of boundaries is Grains method

0.4 (20 Jun 2016)
-----------------
* Sample neighbors_dist method to calculate neighbors distances
* Grains and Boundaries nndist to calculate nearest neighbors distances
* Fancy indexing with slices fixed
* Affine transformations affine_transform, rotate, scale, skew, translate
  methods implemented for Grains and Boundaries
* Sample name atribute added
* Sample bids method to get boundary id's related to grain added

0.4.1 (20 Jun 2016)
~~~~~~~~~~~~~~~~~~~
* Examples added to distribution

0.4.2 (02 Sep 2016)
~~~~~~~~~~~~~~~~~~~
* Sample has pairs property(dictionary) to map boundary id to grains id
* Sample triplets method returns list of grains id creating triple points

0.4.3 (02 Sep 2016)
~~~~~~~~~~~~~~~~~~~
* IPython added to requirements

0.4.4 (12 Jan 2017)
~~~~~~~~~~~~~~~~~~~
* Added MAEE (minimum area enclosing ellipse) to grain shape methods
* Removed embedded IPython and IPython requirements

0.4.5 (12 Jan 2017)
~~~~~~~~~~~~~~~~~~~
* shell script ipolylx opens interactive console

0.4.6 (04 Mar 2017)
~~~~~~~~~~~~~~~~~~~
* added plots module (initial)
* representative_point for Grains implemented
* moments calculation including holes
* surfor and parror functions added
* orientation of polygons is unified and checked
* minbox shape method added

0.4.8 (04 Mar 2017)
~~~~~~~~~~~~~~~~~~~
* bugfix

0.4.9 (XX YYY 2017)
~~~~~~~~~~~~~~~~~~~
* getindex method of Grains and Boundaries implemented
* Grain cdist property return centroid-vertex distance function
* Grain cdir property return centroid-vertex direction function
* Grain shape_vector property returns normalized Fourier descriptors
* Grain regularize method returns Grain with regularly distributed vertices