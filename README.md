# PolyLX - python package to visualize and analyze digitized 2D microstructures

[![GitHub version](https://badge.fury.io/gh/ondrolexa%2Fpolylx.svg)](https://badge.fury.io/gh/ondrolexa%2Fpolylx)
[![Build Status](https://travis-ci.org/ondrolexa/polylx.svg?branch=master)](https://travis-ci.org/ondrolexa/polylx)
[![Documentation Status](https://readthedocs.org/projects/polylx/badge/?version=stable)](https://polylx.readthedocs.io/en/stable/?badge=stable)
[![DOI](https://zenodo.org/badge/30773592.svg)](https://zenodo.org/badge/latestdoi/30773592)

## Installation

### PyPI

To install PolyLX, just execute
```
pip install polylx
```
Alternatively, you download the package manually from the Python Package Index [https://pypi.org/project/polylx](https://pypi.org/project/polylx), unzip it, navigate into the package, and use the command:
```
python setup.py install
```
#### Upgrading via pip

To upgrade an existing version of PolyLX from PyPI, execute
```
pip install polylx --upgrade --no-deps
```
Please note that the dependencies (Matplotlib, NumPy, Pandas, NetworkX, seaborn, shapely, pyshp and SciPy) will also be upgraded if you omit the `--no-deps` flag; use the `--no-deps` ("no dependencies") flag if you don't want this.

#### Installing PolyLX from the source distribution

In rare cases, users reported problems on certain systems with the default pip installation command, which installs PolyLX from the binary distribution ("wheels") on PyPI. If you should encounter similar problems, you could try to install PolyLX from the source distribution instead via
```
pip install --no-binary :all: polylx
```
Also, I would appreciate it if you could report any issues that occur when using `pip install polylx` in hope that we can fix these in future releases.

### Master version

The PolyLX version on PyPI may always one step behind; you can install the latest development version from the GitHub repository by executing
```
pip install git+git://github.com/ondrolexa/polylx.git
```
Or, you can fork the GitHub repository from [https://github.com/ondrolexa/polylx](https://github.com/ondrolexa/polylx) and install PolyLX from your local drive via
```
python setup.py install
```

## Getting started

Documentation is in progress, but you can see PolyLX in action in accompanied Jupyter notebook
[https://nbviewer.ipython.org/github/ondrolexa/polylx/blob/master/polylx_tutorial.ipynb](https://nbviewer.ipython.org/github/ondrolexa/polylx/blob/master/polylx_tutorial.ipynb)

## Documentation

Explore the full features of APSG. You can find detailed documentation [here](https://polylx.readthedocs.org).

## Contributing

Most discussion happens on [Github](https://github.com/ondrolexa/polylx). Feel free to open [an issue](https://github.com/ondrolexa/polylx/issues/new) or comment on any open issue or pull request. Check ``CONTRIBUTING.md`` for more details.

## License

APSG is free software: you can redistribute it and/or modify it under the terms of the MIT License. A copy of this license is provided in ``LICENSE`` file.
