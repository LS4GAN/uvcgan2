#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'uvcgan2',
    version          = '0.0.1',
    author           = 'The LS4GAN Project Developers',
    author_email     = 'dtorbunov@bnl.gov',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = "UVCGAN2 reference implementation",
    packages         = setuptools.find_packages(
        include = [ 'uvcgan2', 'uvcgan2.*' ]
    ),
    install_requires = [ 'numpy', 'pandas', 'tqdm', 'Pillow' ],
)

