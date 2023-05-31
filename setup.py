#! /usr/bin/env python

from setuptools import setup
from setuptools import find_packages


try:
    with open('README.md', 'r') as f:
        readme = f.read()
except Exception:
    readme = str('')


install_requires = [
    'einops',
    'numpy',
    'matplotlib',
    'opencv-python',
    'pillow',
    'torch >= 1.8',
    'torchvision',
    'tqdm',
    'pytest',
    'timm >= 0.6, <=0.6.13',
]


setup(
    name='detectron2_timm',
    version='0.0.1',
    long_description=readme,
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
    test_suite='tests',
)
