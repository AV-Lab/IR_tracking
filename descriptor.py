#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 01:50:39 2023

@author: nadya
"""

import numpy as np
import cv2 as cv
from skimage.feature import hog
from kymatio.sklearn import Scattering2D
import math
        
def compute_descriptor(frame):
    
    # Compute the HOG descriptor
    d = min(frame.shape[0], frame.shape[1])
    d = int(d/2)
    #d = int(math.sqrt(frame.shape[0] * frame.shape[1] / 40))
    #if d < 8: d = 8
    hog_descriptor = hog(frame, orientations=5, pixels_per_cell=(d, d), cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
    hog_descriptor = [[h] for h in hog_descriptor]
    
    return hog_descriptor
