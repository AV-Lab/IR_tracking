#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:06:54 2023

@author: nadya
"""

import cv2
from descriptor import compute_descriptor

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_boxes(image, detections):
    colors = COLORS * 100
    for d in detections:
        b = d[0]
        xmin = int(b[0].item())
        ymin = int(b[1].item())
        xmax = int(b[2].item())
        ymax = int(b[3].item())
        label = d[1]
        score = d[2].item()
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)  # Green color, thickness = 2
        cv2.putText(image, label, (xmax, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Label text
    return image



def process_detections(frame, detections):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height` + descriptor """
    
    detections_p = []
    for d in detections:
        b = d[0]
        xmin = int(b[0].item())
        ymin = int(b[1].item())
        xmax = int(b[2].item())
        ymax = int(b[3].item())
        patch = frame[ymin:ymax, xmin:xmax]
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        desc = compute_descriptor(patch)
        
        w = xmax - xmin
        x = w/2 + xmin
        h = ymax - ymin
        y = h/2 + ymin
        a = w/h
        
        detections_p.append(((x,y,a,h),desc))
    return detections_p