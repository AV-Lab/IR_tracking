# Multi-Target Tracker for Low Light Vision

### Introduction

This repository contains implementation for Multi-Target Tracker for Low Light Vision. LLV tracker is a multi-object tracker for TIR images with a focus on simple and
real-time efficient algorithmic solution. We base our solution on DeepSORT algorithm and extend it to TIR tracking of both, pedestrians and vehicles. To adopt DeepSORT tracker,
we design an appearance descriptor suitable for association problem of TIR images. Furthermore, to address the problem of missing association and detection, we propose a fusion block to
merge short tracklets belonging to the same object in one track. We evaluate the tracker on CAMEL dataset and experimentally on the sequences we collected using an IR-camera.

### Running the tracker
``` 
python deep_sort_app.py \
    --sequence_dir=./MOT16/test/MOT16-06 \
    --detection_file=./resources/detections/MOT16_POI_test/MOT16-06.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True
``` 

### The main entry point is in run_tracking.py. This file runs the tracker on a specified sequence.

In package deep_sort is the main tracking code:

detection.py: Detection base class.
kalman_filter.py: A Kalman filter implementation and concrete parametrization for image space filtering.
linear_assignment.py: This module contains code for min cost matching and the matching cascade.
iou_matching.py: This module contains the IOU matching metric.
nn_matching.py: A module for a nearest neighbor matching metric.
track.py: The track class contains single-target track data such as Kalman state, number of hits, misses, hit streak, associated feature vectors, etc.
tracker.py: This is the multi-target tracker class.
The deep_sort_app.py expects detections in a custom format, stored in .npy files. These can be computed from MOTChallenge detections using generate_detections.py. We also provide pre-generated detections.


### Citing LLV Tracker
If you find this repo useful in your research, please consider citing the following paper:
