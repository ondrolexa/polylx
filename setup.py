#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md')) as f:
    readme = f.read()

with open(path.join(this_directory, 'HISTORY.md')) as f:
    history = f.read()

requirements = [
    'numpy',
    'matplotlib',
    'pandas',
    'seaborn',
    'networkx',
    'scipy',
    'shapely',
    'pyshp'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='polylx',
    version='0.5.0',
    description="A Python package to visualize and analyze microstructures.",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    author="Ondrej Lexa",
    author_email='lexa.ondrej@gmail.com',
    url='https://github.com/ondrolexa/polylx',
    packages=find_packages(),
    package_data={'polylx': ['example/*.*']},
    include_package_data=True,
    install_requires=requirements,
    entry_points="""
    [console_scripts]
    ipolylx=polylx.shell:main
    """,
    license="MIT",
    zip_safe=False,
    keywords='polylx',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
