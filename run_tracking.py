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
from utils_functions import plot_boxes, process_detections
from tracker import Tracker

def run(sequence_dir, ground_truth, output_file, detector_weights, min_confidence, device):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    ground_truth : str
        Path to the ground truth file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.

    """ 
 
    detector = Detector(detector_weights, min_confidence, device)    
    tracker = Tracker()
    files = glob.glob('{}/*'.format(sequence_dir))
    files = sorted(files)
    results = []
    
    print('Total to process : {}'.format(len(files)))

    for idx, f in enumerate(files):
        print("Processing frame {}".format(idx))
        
        save_path = f.split('.')[0] + 'tracked.jpg'
        frame = cv2.imread(f)
        imgsz = frame.shape[1]
        detections = detector.detect(f, imgsz) # in the format xmin, ymin, xmax, ymax
        
        # covert detections to the format x,y,a,h,descriptor
        detections = process_detections(frame, detections)
        #frame = plot_boxes(frame, detections)
        
        #print(detections)

        # Update tracker
        tracker.predict()
        tracker.update(detections)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            xmin, ymin, xmax, ymax = track.to_tlbr() 
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)  # Green color, thickness = 2
            cv2.putText(frame, str(track.track_id), (int(xmin) + 2, int(ymin) + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Label text
        
        cv2.imwrite(save_path, frame)
        # append to results
        # to do



if __name__ == "__main__":
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Sparse Track")
    parser.add_argument("--sequence_dir", help="Path to sequence directory", 
                        default='data/PTB-TIR/classroom3/img')
    parser.add_argument("--ground_truth", help="Path to the tracking ground truth file", 
                        default='data/PTB-TIR/classroom3/anno')
    parser.add_argument("--output_file", help="Path to the tracking output file", 
                        default="/tmp/hypotheses.txt")
    parser.add_argument("--detector_weights", help="Path to detector weights", 
                        default='yolov7.pt')
    parser.add_argument("--min_confidence", help="Disregard detections lower threshold", 
                        default=0.75, type=float)
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    
    run(args.sequence_dir, args.ground_truth, args.output_file, args.detector_weights, args.min_confidence, args.device)