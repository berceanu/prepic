#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "unyt>=2.3.1",
    "numpy>=1.16.1",
    "scipy>=1.3.1",
    "matplotlib>=3.1.0",
    "matplotlib-label-lines>=0.3.6",
    "sliceplots>=0.3.1",
]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest"]

setup(
    author="Andrei Berceanu",
    author_email="andreicberceanu@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3.6",
    ],
    description="analytically estimate various laser-plasma parameters for experiments and PIC simulations",
    install_requires=requirements,
    python_requires=">= 3.6",
    license="BSD license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="prepic",
    name="prepic",
    packages=find_packages(include=["prepic"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/berceanu/prepic",
    version="0.2.4",
    zip_safe=False,
)
