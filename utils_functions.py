#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:06:54 2023

@author: nadya
"""

import cv2
from descriptor import compute_descriptor
from fastdtw import fastdtw
import numpy as np
from scipy.spatial.distance import cosine
import torch
import motmetrics as mm
import pandas as pd


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_boxes(image, detections):
    colors = COLORS * 100
    for d in detections:
        b = d[0]
        xmin, ymin, xmax, ymax = from_xywh_to_tlbr(b) 
        score = d[1]
        label = d[2]
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)  # Green color, thickness = 2
        cv2.putText(image, label, (xmax, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Label text
    return image



def process_detections(frame, detections):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height` + descriptor """
    
    detections_p = []
    for d in detections:
        xmin, ymin, xmax, ymax = from_xywh_to_tlbr(d[0]) 
        x, y, a, h = from_xywh_to_xyah(d[0])
        patch = frame[ymin:ymax, xmin:xmax]
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        desc = compute_descriptor(patch)
        
        detections_p.append(((x,y,a,h),desc,d[1], (xmin, ymin, xmax, ymax)))
    return detections_p

def from_xywh_to_xyah(box):
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    a = w/h
    x = w/2 + x
    y = h/2 + y

    return (x, y, a, h)



def from_xyah_to_tlbr(box):
    """ COnvert to `(min x, miny, max x, max y)`"""

    x = box[0]
    y = box[1]
    a = box[2]
    h = box[3]

    w = a*h
    x_min = x - w/2
    y_min = y - h/2
    x_max = x + w/2
    y_max = y + h/2

    return (x_min, y_min, x_max, y_max)

def read_detections(detections_file):
    detections = {}
    with open(detections_file, 'r') as fin:
        content = fin.read()
        lines = content.strip().split('\n')
        for line in lines:
            els = line.split()
            frame_id = int(els[0])
            dd =  [(int(els[2]), int(els[3]), int(els[4]), int(els[5])), float(els[6]), els[7]]
            if frame_id in detections:
                detections[frame_id].append(dd)
            else:
                detections[frame_id] = [dd]
    return detections


def from_tlbr_to_xywh(box):
    """ COnvert to `(x, y, w, h)`"""

    x_min = box[0]
    y_min = box[1]
    x_max = box[2]
    y_max = box[3]

    x = x_min
    y = y_min
    w = x_max-x_min
    h = y_max-y_min

    return (x, y, w, h)

def from_xywh_to_tlbr(box):
    """ COnvert to `(xmin, ymin, xmax, ymax)`"""
    return (box[0], box[1], box[0]+box[2], box[1]+box[3])

def compute_orientation_vector(points):
    mean_point = np.mean(points, axis=0)
    centered_points = points - mean_point
    covariance_matrix = np.dot(centered_points.T, centered_points) / (len(points) - 1)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    orientation_vector = eigenvectors[:, np.argmax(eigenvalues)]

    return orientation_vector


def compute_direction_vector(points):
    # Ensure the input contains at least two points
    if len(points) < 2:
        raise ValueError("At least two points are required to compute the direction vector.")

    # Compute the direction vector
    first_point = np.array(points[0])
    last_point = np.array(points[-1])
    direction_vector = last_point - first_point
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    return direction_vector

def compute_overlap(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # Check for non-overlapping bounding boxes
    if x_max_inter < x_min_inter or y_max_inter < y_min_inter:
        return 0

    # Calculate the area of the intersection rectangle
    intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)

    # Calculate the areas of each bounding box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the overlap ratio (intersection over union)
    overlap = intersection_area / (box1_area + box2_area - intersection_area)
    return overlap


def evaluation(tSource, gtSource):
  
    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=',')

    # load tracking output
    t = np.loadtxt(tSource, delimiter=',')

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:,0].max())):
        frame += 1
        gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
        t_dets = t[t[:,0]==frame,1:6] # select all detections in t

        C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                    max_iou=0.5) # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:,0].astype('int').tolist(), \
                t_dets[:,0].astype('int').tolist(), C)

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                        'recall', 'precision', 'num_objects', \
                                        'mostly_tracked', 'partially_tracked', \
                                        'mostly_lost', 'num_false_positives', \
                                        'num_misses', 'num_switches', \
                                        'num_fragmentations', 'mota', 'motp' \
                                        ], \
                        name='acc')

    strsummary = mm.io.render_summary(
        summary,
        #formatters={'mota' : '{:.2%}'.format},
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                'precision': 'Prcn', 'num_objects': 'GT', \
                'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
                'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
                'num_misses': 'FN', 'num_switches' : 'IDsw', \
                'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
                }
    )
    print(strsummary)