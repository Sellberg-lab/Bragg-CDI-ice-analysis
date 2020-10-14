# Bragg-CDI-ice-analysis
Bragg CDI analysis of hexagonal ice, including peak finding and peak statistics

-----------------------------
## Installation

Note: `master` branch is currently using python 2.7! To make a virtual environment of python 2 if your default is python 3:

`mkvirtualenv --python=/usr/local/bin/python2 bragg_ice --system-site-packages`

`workon bragg_ice`

`pip install numpy opencv-python==4.2.0.32 matplotlib h5py pandas scipy photutils jupyter`

-----------------------------
## Running the scripts

Inspect ice data, check all options using the help flag `-h`. Example usage follows below.

* inspect only mode (recommended), use a mask to make bad pixels gray, set the upper intensity limit to 5000 ADU, output more information to screen:

`./inspectIce.py -r 0144 -i -m r0144_type0-pixel_mask_r0166_borders+auto+variance+gain.h5 -M 5000 -v`
