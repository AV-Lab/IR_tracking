#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 13:15:21 2023

@author: nadya
"""

import argparse
import os
import cv2
import numpy as np
import glob

from detector import Detector
from utils_functions import plot_boxes, process_detections, from_xyah_to_tlbr, from_tlbr_to_xywh, read_detections, evaluation
from tracker import Tracker
import motmetrics as mm
import shutil
import tensorflow as tf

categories = {'bike', 'person', 'car', 'motor', 'other vehicle', 'truck', 'rider', 'bus', 'train', 'scooter'}

def run(sequence_dir, min_confidence, device, k):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to sequence directory.
    ground_truth : str
        Path to the ground truth file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.

    """ 
 
    tracker = Tracker(k)
    files = glob.glob('{}/img/*'.format(sequence_dir))
    files = sorted(files)
    results = []

    print('Total to process : {}'.format(len(files)))

    dd = sequence_dir + '/detections/'
    dt = sequence_dir + '/tracking/'
    for d in [dt, dd]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)


    detections = read_detections('{}/detections.txt'.format(sequence_dir))
    tracking_file = '{}/tracking.txt'.format(sequence_dir)
    gt_file = '{}/gt.txt'.format(sequence_dir)

    with open(tracking_file, 'w') as fout:
        for idx, f in enumerate(files):
            #print("Processing frame {}".format(idx))

            if (idx+1) not in detections:
                continue

            frame = cv2.imread(f)
            imgsz = 640 
            frame_detections = detections[idx+1]
            
            # covert detections to the format x,y,a,h,descriptor
            frame_ = np.copy(frame)
            frame_ = plot_boxes(frame_, frame_detections)
            cv2.imwrite(dd + str(idx) + '.jpg', frame_)
            

            frame_detections_p = process_detections(frame, frame_detections)


            # Update tracker
            tracker.predict()
            tracker.update(frame_detections_p)

            # Display tracking only confirmed tracks
            for track in tracker.tracks:
                if track.is_confirmed() and track.time_since_update <= k:
                    xmin, ymin, xmax, ymax = track.to_tlbr() 
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)  # Green color, thickness = 2
                    cv2.putText(frame, "Track " + str(track.track_id), (int(xmin) - 3, int(ymin) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)  # Label text
                    x, y, w, h = track.to_tlwh()
                    fout.write('{}, {}, {}, {}, {}, {}\n'.format(idx+1, track.track_id, x, y, w, h))# left, top, width, height
            
            cv2.imwrite(dt + str(idx) + '.jpg', frame)

    evaluation(tracking_file, gt_file)  
            

if __name__ == "__main__":
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Sparse Track")
    parser.add_argument("--sequence_dir", help="Path to sequence directory", 
                        default='data/City_IR/intersection2') # with images stored in img folder and
                                                         # detections stored in detections.txt 
    parser.add_argument("--min_confidence", help="Disregard detections lower threshold", 
                        default=0.5, type=float)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--k', default=3, help='number of consequitive frames to count track as active')
    args = parser.parse_args()
    
    run(args.sequence_dir, args.min_confidence, args.device, args.k)