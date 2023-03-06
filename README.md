# PolyLX - python package to visualize and analyze digitized 2D microstructures

[![PyPI version](https://badge.fury.io/py/polylx.svg)](https://badge.fury.io/py/polylx)
[![Testing](https://github.com/ondrolexa/polylx/actions/workflows/pythontest.yml/badge.svg?event=push)](https://github.com/ondrolexa/polylx)
[![Documentation Status](https://readthedocs.org/projects/polylx/badge/?version=stable)](https://polylx.readthedocs.io/en/stable/?badge=stable)
[![DOI](https://zenodo.org/badge/30773592.svg)](https://zenodo.org/badge/latestdoi/30773592)

## Installation

### PyPI

To install PolyLX, just execute
```
pip install polylx
```

#### Upgrading via pip

To upgrade an existing version of PolyLX from PyPI, execute
```
pip install polylx --upgrade --no-deps
```
Please note that the dependencies (Matplotlib, NumPy, Pandas, NetworkX, seaborn, shapely, pyshp and SciPy) will also be upgraded if you omit the `--no-deps` flag; use the `--no-deps` ("no dependencies") flag if you don't want this.

#### Installing PolyLX with conda or mamba

Another common way to install is create environment using conda or mamba. Download latest version of [polylx](https://github.com/ondrolexa/polylx/archive/refs/heads/master.zip) and unzip to folder of your choice. Use conda or mamba to create an environment from an ``environment.yml`` file. Open the terminal, change directory where you unzip the source and execute following command:

```
conda env create -f environment.yml
```
Activate the new environment and install from current directory::

```
conda activate polylx
pip install polylx
```

## Getting started

Documentation is in progress, but you can see PolyLX in action in accompanied Jupyter notebook
[https://nbviewer.ipython.org/github/ondrolexa/polylx/blob/master/polylx_tutorial.ipynb](https://nbviewer.ipython.org/github/ondrolexa/polylx/blob/master/polylx_tutorial.ipynb)

## Documentation

Explore the full features of PolyLX. You can find detailed documentation [here](https://polylx.readthedocs.org).

## Contributing

Most discussion happens on [Github](https://github.com/ondrolexa/polylx). Feel free to open [an issue](https://github.com/ondrolexa/polylx/issues/new) or comment on any open issue or pull request. Check ``CONTRIBUTING.md`` for more details.

## License

PolyLX is free software: you can redistribute it and/or modify it under the terms of the MIT License. A copy of this license is provided in ``LICENSE`` file.
