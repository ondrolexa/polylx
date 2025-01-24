# PolyLX - python package to visualize and analyze digitized 2D microstructures

[![PyPI version](https://badge.fury.io/py/polylx.svg)](https://badge.fury.io/py/polylx)
[![Testing](https://github.com/ondrolexa/polylx/actions/workflows/pythontest.yml/badge.svg?event=push)](https://github.com/ondrolexa/polylx)
[![Documentation Status](https://readthedocs.org/projects/polylx/badge/?version=stable)](https://polylx.readthedocs.io/en/stable/?badge=stable)
[![DOI](https://zenodo.org/badge/30773592.svg)](https://zenodo.org/badge/latestdoi/30773592)

## Installation

### PyPI

To install PolyLX, create virtual environment, activate it and install with pip:
```
python -m venv polylx
source polylx/bin/activate
pip install polylx
```

#### Upgrading via pip

To upgrade an existing version of PolyLX from PyPI, execute:
```
pip install polylx --upgrade
```

### Installing PolyLX with mamba

Another common way to install is create environment using mamba (or conda with conda-forge repository):

```
mamba create -n polylx numpy matplotlib scipy pandas pyarrow seaborn networkx shapely pyshp fiona jupyterlab pyefd jenkspy shapelysmooth
```

Activate the new environment and install with pip:

```
mamba activate polylx
pip install polylx
```

## Documentation

Explore the full features of PolyLX. You can find detailed documentation [here](https://polylx.readthedocs.org).

## Contributing

Most discussion happens on [Github](https://github.com/ondrolexa/polylx). Feel free to open [an issue](https://github.com/ondrolexa/polylx/issues/new) or comment on any open issue or pull request. Check ``CONTRIBUTING.md`` for more details.

## License

PolyLX is free software: you can redistribute it and/or modify it under the terms of the MIT License. A copy of this license is provided in ``LICENSE`` file.
