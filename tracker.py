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
from scipy import spatial
from skimage.feature import match_template
from utils_functions import from_xyah_to_tlbr, compute_orientation_vector, compute_direction_vector, compute_overlap
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from sklearn.metrics.pairwise import cosine_similarity




INFTY_COST = 1e+15
gated_cost=INFTY_COST
gating_threshold = 100 #16.919 #9.4877

class Tracker:


    def __init__(self, max_age=50, k=5):
        self.max_age = max_age
        self.n_init = k
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.max_iou_distance = 0.5
        

    def predict(self):
        """Propagate track state distributions one time step forward."""
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management."""
        
        matches, unmatched_tracks, unmatched_detections = self.match(detections)

        #Associate remaining tracks together with unmatched detections using IOU.
        iou_track_candidates = [k for k in unmatched_tracks if self.tracks[k].time_since_update <= 1]
        unmatched_tracks = [k for k in unmatched_tracks if self.tracks[k].time_since_update > 1]
        matches_iou, unmatched_tracks_iou, unmatched_detections = self.compute_iou_matches(detections, 
                                                                                            iou_track_candidates, 
                                                                                             unmatched_detections)
        matches += matches_iou
        unmatched_tracks = list(set(unmatched_tracks + unmatched_tracks_iou))

        # For each match update with observations
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
            
        # For each track that does not have match mark missed     
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # For each unmatch detection initiate a new track
        for detection_idx in unmatched_detections:
            self.initiate_track(detections[detection_idx])

        #Perform track fusion between confirmed tracks age more than k and tracks which just became confirmed
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.age > self.n_init and t.is_confirmed()]
        tracks_to_fuse = [i for i,t in enumerate(self.tracks) if t.age == self.n_init]
        self.fuse_tracks(confirmed_tracks, tracks_to_fuse)

        # Remove tracks marked deleted
        self.tracks = [t for t in self.tracks if not t.is_deleted()]


    def initiate_track(self, detection):
        """Creating a new track."""
        mean, covariance = self.kf.initiate(detection[0])
        self.tracks.append(Track(mean, covariance, self._next_id, 
                                 self.n_init, self.max_age, detection[1]))
        self._next_id += 1
        
    def match(self, detections):   
        """Perform global matching """
        
        track_indices = list(range(len(self.tracks)))
        detection_indices = list(range(len(detections)))
        matches, unmatched_tracks, unmatched_detections = self.min_cost_matching(detections, 
                                                                                track_indices, 
                                                                                detection_indices)   
        return matches, unmatched_tracks, unmatched_detections

                
    def min_cost_matching(self, detections, track_indices, detection_indices):
        """Solve linear assignment problem for matching tracks and detctions."""
        
        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices  # Nothing to match.

        cost_matrix  = self.compute_cost_matrix(detections, track_indices, detection_indices)
        indices = self.majority_voting(cost_matrix)
        matches, unmatched_tracks, unmatched_detections = [], [], []
        
        # Regsiter dectections that were not matched after Hungarian algorithm
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[1]:
                unmatched_detections.append(detection_idx)
                
        # Register tracks with no matching detection
        for row, track_jdx in enumerate(track_indices):
            if row not in indices[0]:
                unmatched_tracks.append(track_jdx)
                
        # Register matches        
        for row, col in zip(indices[0], indices[1]):
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            matches.append((track_idx, detection_idx))
                
        return matches, unmatched_tracks, unmatched_detections
    
    def compute_cost_matrix(self, detections, track_indices, detection_indices):
        """Computing Cost matrix."""
                
        features = [detections[i][1] for i in detection_indices]
        targets = [self.tracks[i].descriptor for i in track_indices]

        cost_matrix = np.zeros((3, len(targets), len(features)))

        for i, t in enumerate(targets):
            for j, d in enumerate(features):
                for idx in range(3):
                    cost_matrix[idx][i][j] = cosine(t[idx], d[idx])

        measurements = np.asarray([detections[i][0] for i in detection_indices])
        

        for row, track_idx in enumerate(track_indices):
           track = self.tracks[track_idx]
           gating_distance = self.kf.gating_distance(track.mean, track.covariance, measurements)
           cost_matrix[:, row, gating_distance > gating_threshold] = gated_cost

        return cost_matrix


    def majority_voting(self, cost_matrix):
        indices = []
        matches = {}
        final_matches = ([],[])

        #print(cost_matrix)

        # Hungarian algorithm
        for idx in range(3):
            x, y = linear_assignment(cost_matrix[idx])
            for row, col in zip(x, y):
                if cost_matrix[0][row][col] != gated_cost:
                    if (row, col) in matches:
                        matches[(row, col)] += 1
                    else: matches[(row, col)] = 1

        for k, v in matches.items():
            if v >= 2:
                final_matches[0].append(k[0])
                final_matches[1].append(k[1])


        #print(final_matches)
        return final_matches

    def compute_iou_matches(self, detections, track_indices, detection_indices):
        """Computing Cost matrix."""
                
        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices  # Nothing to match.

        features = [detections[i][3] for i in detection_indices]
        targets = [self.tracks[i].to_tlbr() for i in track_indices]

        cost_matrix = np.zeros((len(targets), len(features)))

        for i, t in enumerate(targets):
            for j, d in enumerate(features):
                iou = compute_overlap(t,d)
                #print(iou)
                cost_matrix[i][j] = iou if iou > self.max_iou_distance else gated_cost

        indices = linear_assignment(cost_matrix)

        matches, unmatched_tracks, unmatched_detections = [], [], []
        
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[1]:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(track_indices):
            if row not in indices[0]:
                unmatched_tracks.append(track_idx)
        for row, col in zip(indices[0], indices[1]):
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if cost_matrix[row, col] == gated_cost:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        return matches, unmatched_tracks, unmatched_detections



    def fuse_tracks(self, confirmed_tracks, tracks_to_fuse):
        """Performs fusion of the tracks."""

        if len(confirmed_tracks) == 0 or len(tracks_to_fuse) == 0:
            return 

        
        cost_matrix = np.zeros((len(confirmed_tracks), len(tracks_to_fuse)))

        h_confirmed_tarcks = [np.array(self.tracks[t].tracking_history[-self.n_init:])[:,:2] for t in confirmed_tracks]
        h_tracks_to_fuse = [np.array(self.tracks[t].tracking_history[-self.n_init:])[:,:2] for t in tracks_to_fuse]

        for i, t in enumerate(h_confirmed_tarcks):
            for j, d in enumerate(h_tracks_to_fuse):
                distance = fastdtw(t,d) 
                cost_matrix[i][j] = distance[0]

        measurements = np.array([self.tracks[t].tracking_history[-1] for t in tracks_to_fuse])

        for row, track_idx in enumerate(confirmed_tracks):
             track = self.tracks[track_idx]
             gating_distance = self.kf.gating_distance(track.mean, track.covariance, measurements)
             for col, gate in enumerate(gating_distance): 
                if gate > gating_threshold:
                    cost_matrix[row, col]= gated_cost


        indices = linear_assignment(cost_matrix)

        # Merge Tracks        
        for row, col in zip(indices[0], indices[1]):
            if cost_matrix[row][col] != gated_cost:
                print(confirmed_tracks[row], tracks_to_fuse[col])
                self.merge_tracks(confirmed_tracks[row], tracks_to_fuse[col])
                print('merged')



    def merge_tracks(self, idx, jdx):
        """Fusing two tracks."""
        self.tracks[idx].hits += self.tracks[jdx].hits
        self.tracks[idx].time_since_update = 0
        self.tracks[idx].descriptor = 0.75*self.tracks[idx].descriptor + 0.25*self.tracks[jdx].descriptor
        self.tracks[jdx].set_deleted()