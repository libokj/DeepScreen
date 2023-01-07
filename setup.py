#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="deepscreen",
    version="0.0.1",
    description="A unified and extensible deep learning toolkit to facilitate rapid drug discovery.",
    author="Bo Li",
    author_email="libokj@live.cn",
    url="https://github.com/libokj/DeepScreen",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
