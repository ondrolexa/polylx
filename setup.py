#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
from setuptools import setup, find_packages

CURRENT_PATH = path.abspath(path.dirname(__file__))

with open(path.join(CURRENT_PATH, 'README.md')) as f:
    readme = f.read()

with open(path.join(CURRENT_PATH, 'HISTORY.md')) as f:
    history = f.read()

setup(
    name='polylx',
    version='0.5.1',
    description="A Python package to visualize and analyze microstructures.",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    author="Ondrej Lexa",
    author_email='lexa.ondrej@gmail.com',
    url='https://github.com/ondrolexa/polylx',
    packages=find_packages(),
    package_data={'polylx': ['example/*.*']},
    include_package_data=True,
    install_requires=['numpy', 'matplotlib', 'pandas', 'seaborn',
                      'networkx', 'scipy', 'shapely', 'pyshp', 'pyefd'],
    extras_require={
        'docs': ['sphinx'],
        'test': ['pytest'],
        'lint': ['black', 'flake8'],
        'jupyter': ['jupyterlab'],
    },
    entry_points="""
    [console_scripts]
    ipolylx=polylx.shell:main
    """,
    license="MIT",
    zip_safe=False,
    keywords='polylx',
    python_requires=">=3.6",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    project_urls={
        'Documentation': 'https://polylx.readthedocs.io/',
        'Source Code': 'https://github.com/ondrolexa/polylx/',
        'Bug Tracker': 'https://github.com/ondrolexa/polylx/issues/',
    },
)
