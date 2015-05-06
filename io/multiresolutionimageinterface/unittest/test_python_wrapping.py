# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:20:37 2015

@author: Geert
"""

import sys
sys.path.append(r"D:\Code\sources\diag\build (VC12)\bin\Release")
import multiresolutionimageinterface
import numpy as np
level = 6
r = multiresolutionimageinterface.MultiResolutionImageReader()
w = multiresolutionimageinterface.MultiResolutionImageWriter()
i = r.open(r"D:\Temp\T06-12822-II1-10.mrxs")
w.openFile(r"D:\Temp\test_python.tif")
w.setTileSize(512)
w.setCompression(multiresolutionimageinterface.LZW)
w.setDataType(multiresolutionimageinterface.UChar)
w.setColorType(multiresolutionimageinterface.RGB)
dims = i.getLevelDimensions(level)
a = np.zeros(512*512*4, dtype='ubyte')
w.writeImageInformation(dims[0], dims[1])
for y in range(0, dims[1], 512):
    print y
    for x in range(0, dims[0], 512):
        i.getUCharPatch(int(x*i.getLevelDownsample(level)),
                        int(y*i.getLevelDownsample(level)), 512, 512, level, a)
        w.writeBaseImagePart(a.reshape(512, 512, 4)[:, :, 2::-1].flatten())
w.finishImage()
