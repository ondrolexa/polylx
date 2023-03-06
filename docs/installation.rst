============
Installation
============

You need to have Python and dependencies installed locally to use **polylx**. Just follow these steps:

  1. Easiest way to install Python is to use `Anaconda <https://www.anaconda.com/distribution>`_/
  `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_/
  `Miniforge <https://github.com/conda-forge/miniforge>`_ distribution.
  Download it and follow installation steps.

  2. Download latest version of `polylx <https://github.com/ondrolexa/polylx/archive/refs/heads/master.zip>`_
  and unzip to folder of your choice.

  3. Use conda or mamba to create an environment from an ``environment.yml``
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

      pip install --upgrade https://github.com/ondrolexa/pypsbuilder/archive/master.zip