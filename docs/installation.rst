============
Installation
============


PyPI
----

To install PolyLX, just execute::

  pip install polylx

Upgrading via pip
-----------------

To upgrade an existing version of PolyLX from PyPI, execute::

  pip install polylx --upgrade --no-deps

Please note that the dependencies (Matplotlib, NumPy, Pandas, NetworkX, seaborn, shapely, pyshp and SciPy) will also be upgraded if you omit the `--no-deps` flag; use the `--no-deps` ("no dependencies") flag if you don't want this.

Installing PolyLX with conda or mamba
-------------------------------------

You need to have Python and dependencies installed locally to use **polylx**. Just follow these steps:

  1. It is suggested to use `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Miniforge <https://github.com/conda-forge/miniforge>`_ installer. Download it and follow the installation steps.

  2. Download latest version of `polylx <https://github.com/ondrolexa/polylx/archive/refs/heads/master.zip>`_
  and unzip to folder of your choice.

  3. Use ``conda`` or ``mamba`` to create an environment from an ``environment.yml``
  file. Open the terminal, change directory where you unzip the source
  and execute following command::

      conda env create -f environment.yml

  4. Activate the new environment and install from current directory::

      conda activate polylx
      pip install polylx

Upgrade to latest version
-------------------------

You can anytime upgrade your existing **polylx** to the latest released version::

          pip install --upgrade polylx

or to latest master version at github::

      pip install --upgrade https://github.com/ondrolexa/polylx/archive/refs/heads/master.zip