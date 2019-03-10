# Bragg-CDI-ice-analysis
Bragg CDI analysis of hexagonal ice, including peak finding and peak statistics

# inspect ice data

* Inspect only mode (recommended), uses a mask to make bad pixels gray, sets the upper intensity limit to 5000 ADU, verbose to output more information to screen:

`./inspectIce.py -r 0144 -i -m r0144_type0-pixel_mask_r0166_borders+auto+variance+gain.h5 -M 5000 -v`

