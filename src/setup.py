#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Setup script."""
from setuptools import find_packages, setup

setup(
    # mandatory
    name="proteinmetrics",
    # mandatory
    version="0.1",
    # mandatory
    author="Philip Hartout, Tim Kucera",
    packages=find_packages(),
    setup_requires=["isort"],
)
