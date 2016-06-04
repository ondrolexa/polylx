#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'numpy',
    'matplotlib',
    'pandas',
    'networkx',
    'shapely'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='polylx',
    version='0.3.2',
    description="A Python package to visualize and analyze microstructures.",
    long_description=readme + '\n\n' + history,
    author="Ondrej Lexa",
    author_email='lexa.ondrej@gmail.com',
    url='https://github.com/ondrolexa/polylx',
    packages=[
        'polylx',
    ],
    package_dir={'polylx':
                 'polylx'},
    include_package_data=True,
    install_requires=requirements,
    scripts=['ipolylx'],
    license="BSD",
    zip_safe=False,
    keywords='polylx',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
