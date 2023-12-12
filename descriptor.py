#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 01:50:39 2023

@author: nadya
"""

import numpy as np
import cv2 as cv
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.feature import daisy
from skimage.feature import graycomatrix, graycoprops
import math


radius = 3
n_points = 8 * radius
l = 324
d = 8
num_orientations = 8
num_scales = 5
min_wavelength = 4
max_wavelength = 32
num_regions = 2
        
def compute_gabor_features(image, num_orientations, num_scales, min_wavelength, max_wavelength):

    gabor_features = []

    for scale in np.linspace(min_wavelength, max_wavelength, num_scales):
        for orientation in range(num_orientations):
            # Generate Gabor kernel
            theta = orientation / num_orientations * np.pi
            kernel = cv.getGaborKernel((int(scale), int(scale)), 1.0, theta, scale, 0.5, 0, ktype=cv.CV_32F)

            # Filter the image using the Gabor kernel
            filtered_image = cv.filter2D(image, cv.CV_32F, kernel)
            gabor_features.append(filtered_image)

    return np.array(gabor_features)

def compute_image_descriptor(gabor_features, num_regions):

    height, width, num_filters = gabor_features.shape

    # Compute the region size
    region_height = height // num_regions
    region_width = width // (num_regions+1)

    image_descriptor = []

    for i in range(num_regions):
        for j in range(num_regions+1):
            # Extract the region from Gabor features
            region = gabor_features[i*region_height:(i+1)*region_height, j*region_width:(j+1)*region_width, :]

            # Compute the mean and variance for each Gabor filter response in the region
            mean_response = np.mean(region, axis=(0, 1))
            variance_response = np.var(region, axis=(0, 1))

            # Concatenate the mean and variance values to form the region descriptor
            region_descriptor = np.concatenate((mean_response, variance_response))
            image_descriptor.extend(region_descriptor)

    return np.array(image_descriptor)


def compute_descriptor(frame):
    
    # Compute HOG descriptor
    dim=(32,32)
    resized = cv.resize(frame, dim, interpolation = cv.INTER_AREA)
    hog_descriptor = hog(resized, orientations=9, pixels_per_cell=(d, d), cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')

    # Compute DAISY descriptor
    descs, descs_img = daisy(resized, step=8, radius=8, rings=3, histograms=12, orientations=9)
    daisy_descriptor = descs[0].ravel()[:l]
    

    gabor_features = compute_gabor_features(resized, num_orientations, num_scales, min_wavelength, max_wavelength)
    image_descriptor = compute_image_descriptor(gabor_features, num_regions)

    image_descriptor = image_descriptor.tolist()
    image_descriptor = image_descriptor[:l]

    descriptor = np.vstack((hog_descriptor, hog_descriptor, hog_descriptor))
    
    
    return descriptor



