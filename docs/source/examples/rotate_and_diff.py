# -*- coding: utf-8 -*-
"""
Align and diff two data sets based on user clicked markers.
libraries download: https://github.com/vincenzooo/pyXTel

INTERACTIVE MARKERS: 
    CTRL + left click: add marker
    CTRL + right click: add marker
    ENTER: continue and return transformation
Window might flicker if there are other windows, 
"""

import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
from dataIO.outliers import remove_outliers
from dataIO.span import span
from plotting.add_clickable_markers import add_clickable_markers2
from plotting.backends import maximize
from plotting.multiplots import plot_difference
from pySurf.affine2D import find_affine
from pySurf.data2D import align_interactive
from pySurf.data2D_class import Data2D
from pySurf.readers.instrumentReader import fitsWFS_reader

"""INPUT SETTINGS"""
plt.ion()
# see https://stackoverflow.com/questions/59596264/defining-package-level-path-constants
# infolder="G:\\My Drive\\materialsLabShared\\WFS_BONDING_TESTS\\181016_PCO2S06_Reproducibility"
print("**",os.path.realpath('.'),"**")  # to understand why it works only from .

infolder =  r"..\..\..\test\data" #r"..\..\..\pySurf\test\input_data\fits\WFS"
outfolder = r"..\..\..\test\results\rotate_and_diff"

file1 = "181016_01_PCO2S06_1009_08.fits"
file2 = "181016_02_PCO2S06_1009_08.fits"
scale = 101.6 / 116  # ratio between mm and pixels
ytox = 220.0 / 200  # aspect ratio of pixel
strip = True  # strip nan frame

interactive = False
""""""
# Replace with updated code
rfiles = [os.path.join(infolder, f) for f in [file1, file1]]
dlist = [
    Data2D(
        *fitsWFS_reader(
            f, scale=(scale, scale, 1), center=(0, 0), strip=strip, ytox=ytox, ypix=1
        ),
        units=["mm", "mm", "um"],
    )
    for f in rfiles
]
# dlist[1]=dlist[1].rotate(1.5) #rotate 1.5 deg for test
os.makedirs(outfolder,exist_ok=True)

dl = [d.level((10, 0)) for d in dlist]
# replace with fixed markers
if interactive:
    mref, mtrans = align_interactive(
        dl
    )  # with default argument return transformation to first coordinates
    #pdb.set_trace()
else:
    mref =  [np.array([[-43.9760178 ,  27.17066963],
                    [ 14.20627364, -10.8494416 ],
                    [-44.55208009, -20.64250056]]), np.array([[-44.29023359,  29.4749188 ],
                    [ 13.31599555,  -7.96913014],
                    [-42.56204672, -19.49037597]])]
    mtrans = [None,find_affine(*mref)]


# work on individual data files (can be replaced by functions in WFS_repeatability to work on N files:
d1, d2 = dlist
d2t = d1.apply_transform(mtrans[1])
maximize()
plt.savefig(os.path.join(outfolder,'alignment.png'))
plt.figure()
diff = plot_difference(d1, d2t)
maximize()

print("Alignment on markers:\n",mref)
plt.savefig(os.path.join(outfolder,'difference.png'))
diff.save(os.path.join(outfolder,'difference.dat'))
print("Figure saved in ",outfolder)
