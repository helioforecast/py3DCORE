# 3DCORE
========

Python implementations of the class of models running under the name of "3D Coronal Rope Ejection Model", a modelling techniqe for coronal mass ejection (CME) flux ropes. Originally forked from https://github.com/ajefweiss/py3DCORE.

## Installation
------------
First install new conda environment:

    conda create -n "3dcorenv" python=3.10.10
    conda activate 3dcorenv
    
Install the latest version of HelioSat manually using `git`:

    git clone https://github.com/ajefweiss/HelioSat
    cd HelioSat
    pip install -e .
    
Install the latest version manually using `git`:

    git clone https://github.com/helioforecast/py3DCORE
    cd py3DCORE
    pip install -e .
    
Install all necessary packages:
    
    pip install -r requirements.txt
    

or the original version from https://github.com/ajefweiss/py3DCORE.

------------

## Notes on 3DCORE Results
------------

3DCORE can work with several coordinate frames. Be careful to interpret results correctly

------------

## Notes on HelioSat
------------

3DCORE uses the package HelioSat (https://github.com/ajefweiss/HelioSat) to retrieve spacecraft data and other spacecraft related information (positions, trajectories, etc.). 

In order for HelioSat to work properly, the following steps are necessary:

1. manually create the folder ~/.heliosat 
2. within .heliosat, manually create the following three folders
    - cache
    - data
    - kernels
3. if HelioSat fails to download kernels, download them manually and place them in the kernel folder

In those folders, HelioSat will download and save the needed spacecraft data and corresponding kernels. 
If you want to use custom data not available online, place the datafile in .heliosat/data and set custom_data = True during fitting.