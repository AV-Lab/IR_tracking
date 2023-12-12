#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 01:47:38 2023

@author: nadya
"""

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------


    Attributes
    ----------


    """

    def __init__(self, mean, covariance, track_id, n_init, max_age, descriptor=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 0
        self.age = 0
        self.time_since_update = 0
        self.state = TrackState.Tentative
        self.descriptor = descriptor
        self.alpha = 0.9
        self._n_init = n_init
        self._max_age = max_age
        self.tracking_history = []


    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step."""
        
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self.tracking_history.append(self.to_xyah())

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature cache."""
        
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection[0], detection[2])
        self.descriptor = self.alpha * self.descriptor + (1-self.alpha) * detection[1]
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init: 
            self.state = TrackState.Confirmed
            
    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def to_xyah(self):
        """Get current position in bounding box format `(center x, center y. aspect ratio, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[0] += ret[2] / 2
        ret[1] += ret[3] / 2
        ret[2] /= ret[3]
        return [ret[0], ret[1], ret[2], ret[3]]
    
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """

        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def set_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        self.state = TrackState.Tentative

    def set_confirmed(self):
        """Returns True if this track is confirmed."""
        self.state = TrackState.Confirmed

    def set_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        self.state = TrackState.Deleted