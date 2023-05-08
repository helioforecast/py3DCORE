## 3DCORE
========

Python implementations of the class of models running under the name of "3D Coronal Rope Ejection Model", a modelling techniqe for coronal mass ejection (CME) flux ropes. Originally forked from https://github.com/ajefweiss/py3DCORE.

# Installation
------------

Install the latest version manually using `git`:

    git clone https://github.com/helioforecast/py3DCORE
    cd py3DCORE
    pip install .

or the original version from https://github.com/ajefweiss/py3DCORE.

------------

# Notes on HelioSat
------------

3DCORE uses the package HelioSat (https://github.com/ajefweiss/HelioSat) to retrieve spacecraft data and other spacecraft related information (positions, trajectories, etc.). 

In order for HelioSat to work properly, the following steps are necessary:

1. manually create the folder .heliosat 
2. within .heliosat, manually create the following three folders
    - cache
    - data
    - kernels

In those folders, HelioSat will download and save the needed spacecraft data and corresponding kernels. 
