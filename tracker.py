#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 00:38:54 2023

@author: nadya
"""

from __future__ import absolute_import
import numpy as np
from kalman_filter import KalmanFilter
from track import Track
from descriptor import compute_descriptor
from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

INFTY_COST = 1e+5
gated_cost=INFTY_COST

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------


    Attributes
    ----------


    """

    def __init__(self, max_distance= 100, max_age=30, n_init=3):
        self.max_distance = max_distance
        self.max_age = max_age
        self.n_init = n_init
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management."""
        
        # Run matching Procedure
        matches, unmatched_tracks, unmatched_detections = self.match(detections)

        # For each match update with observations
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
            
        # For each track that does not have match mark missed     
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # For each unmatch detection initiate a new track
        for detection_idx in unmatched_detections:
            self.initiate_track(detections[detection_idx])
            
        # Remove tracks marked deleted
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def initiate_track(self, detection):
        """Creating a new track."""
        mean, covariance = self.kf.initiate(detection[0])
        self.tracks.append(Track(mean, covariance, self._next_id, 
                                 self.n_init, self.max_age, detection[1]))
        self._next_id += 1
        
    def match(self, detections):   
        """Perform matching cascade."""
        
        cascade_depth = self.max_age
        track_indices = list(range(len(self.tracks)))
        detection_indices = list(range(len(detections)))

        # before matching all detections are unmatched, we store indicies
        unmatched_detections = detection_indices
        matches = []
        
        # The matching is happening in a cascade form: at each age level 
        # we match detections that remained unmatched at the previous level
        
        for level in range(cascade_depth):
            if len(unmatched_detections) == 0:  # No detections left
                break

            # we select tracks with age = level 
            track_indices_l = [
                k for k in track_indices
                if self.tracks[k].time_since_update == 1 + level
            ]
            if len(track_indices_l) == 0:  # Nothing to match at this level
                continue

            matches_l, _, unmatched_detections = self.min_cost_matching(detections,
                                                                   track_indices_l, 
                                                                   unmatched_detections)
            matches += matches_l
            
        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
        return matches, unmatched_tracks, unmatched_detections

                
    def min_cost_matching(self, detections, track_indices, unmatched_detection_indices):
        """Solve linear assignment problem for matching tracks and detctions."""
        
        if len(unmatched_detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, unmatched_detection_indices  # Nothing to match.

        cost_matrix = self.compute_cost_matrix(detections, track_indices, unmatched_detection_indices)
        cost_matrix[cost_matrix > self.max_distance] = self.max_distance + 1e-5
        
        # Hungarian algorithm
        indices = linear_assignment(cost_matrix)
        
        print(indices)

        matches, unmatched_tracks, unmatched_detections = [], [], []
        
        # Regsiter dectections that were not matched after Hungarian algorithm
        for col, detection_idx in enumerate(unmatched_detection_indices):
            if col not in indices[1]:
                unmatched_detections.append(detection_idx)
                
        # Register tacks with no matching detection
        for row, track_jdx in enumerate(track_indices):
            if row not in indices[0]:
                unmatched_tracks.append(track_jdx)
                
        # Register matches        
        for row, col in zip(indices[0], indices[1]):
            track_idx = track_indices[row]
            detection_idx = unmatched_detection_indices[col]
            if cost_matrix[row, col] > self.max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
                
        return matches, unmatched_tracks, unmatched_detections
    
    def compute_cost_matrix(self, detections, track_indices, unmatched_detection_indices):
        """Computing Cost matrix."""
                
        features = [detections[i][1] for i in unmatched_detection_indices]
        targets = [self.tracks[i].descriptor for i in track_indices]
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, t in enumerate(targets):
            for j, d in enumerate(features):
                #print(len(self.tracks[i].descriptor), len(detections[j][1]))
                distance, path = fastdtw(t, d, dist=euclidean)
                cost_matrix[i][j] = distance


        """Invalidate infeasible entries in cost matrix based on the state
        #distributions obtained by Kalman filtering."""
        
        gating_threshold = 9.4877
        measurements = np.asarray([detections[i][0] for i in unmatched_detection_indices])
        for row, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            gating_distance = self.kf.gating_distance(
                track.mean, track.covariance, measurements)
            cost_matrix[row, gating_distance > gating_threshold] = gated_cost
         
        #print(cost_matrix)
        return cost_matrix

