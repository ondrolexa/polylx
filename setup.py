#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'numpy',
    'matplotlib',
    'pandas',
    'networkx',
    'scipy',
    'shapely'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='polylx',
    version='0.4.5',
    description="A Python package to visualize and analyze microstructures.",
    long_description=readme + '\n\n' + history,
    author="Ondrej Lexa",
    author_email='lexa.ondrej@gmail.com',
    url='https://github.com/ondrolexa/polylx',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
    'console_scripts': [
        'ipolylx=polylx.shell:main'
        ]
    },
    license="BSD",
    zip_safe=False,
    keywords='polylx',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
