"""
Constant-velocity Kalman filter for 2D tracking.

State vector: [x, y, vx, vy]
Each track maintains its own filter. The tracker (Kalman naive OR JPDA) is
responsible for ASSOCIATING measurements to tracks — this file only handles
the math for one track at a time.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import itertools

from ..sensors.sensors import Measurement


_track_id = itertools.count(1)


@dataclass
class Track:
    id: int = field(default_factory=lambda: next(_track_id))
    x: np.ndarray = field(default_factory=lambda: np.zeros(4))  # [x, y, vx, vy]
    P: np.ndarray = field(default_factory=lambda: np.eye(4) * 100.0)  # covariance
    hits: int = 0                  # total updates received
    misses: int = 0                # consecutive ticks without a measurement
    age: float = 0.0               # seconds since created
    confirmed: bool = False        # promoted from tentative to confirmed
    last_sensors: List[str] = field(default_factory=list)  # which sensors updated most recently
    drone_type_votes: dict = field(default_factory=dict)   # camera type-guess histogram

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2]

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    @property
    def heading(self) -> float:
        return float(np.degrees(np.arctan2(self.vel[1], self.vel[0])))


# Process noise — how much we expect drones to randomly accelerate per tick
def _Q(dt: float, accel_std: float = 3.0) -> np.ndarray:
    q = accel_std ** 2
    return np.array([
        [dt**4/4*q,  0,          dt**3/2*q,  0],
        [0,          dt**4/4*q,  0,          dt**3/2*q],
        [dt**3/2*q,  0,          dt**2*q,    0],
        [0,          dt**3/2*q,  0,          dt**2*q],
    ])


def _F(dt: float) -> np.ndarray:
    """State transition: constant velocity."""
    return np.array([
        [1, 0, dt, 0],
        [0, 1, 0,  dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ])


# Measurement model: H projects state -> position (sensors give us position)
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
])


def predict(track: Track, dt: float) -> None:
    """Predict step — propagates the state forward in time."""
    F = _F(dt)
    Q = _Q(dt)
    track.x = F @ track.x
    track.P = F @ track.P @ F.T + Q
    track.age += dt


def update(track: Track, meas: Measurement) -> None:
    """Standard Kalman update with one measurement."""
    z = meas.pos
    R = meas.cov  # measurement noise covariance

    y = z - H @ track.x                     # innovation
    S = H @ track.P @ H.T + R               # innovation covariance
    K = track.P @ H.T @ np.linalg.inv(S)    # Kalman gain

    track.x = track.x + K @ y
    track.P = (np.eye(4) - K @ H) @ track.P

    track.hits += 1
    track.misses = 0
    if meas.sensor not in track.last_sensors:
        track.last_sensors = (track.last_sensors + [meas.sensor])[-4:]

    # Track type from camera votes
    tg = meas.extra.get("type_guess")
    if tg:
        track.drone_type_votes[tg] = track.drone_type_votes.get(tg, 0) + 1


def update_weighted(track: Track, meas_list: List[Measurement], betas: List[float]) -> None:
    """
    JPDA-style weighted update: combines multiple candidate measurements,
    each weighted by its association probability beta.

    Uses the standard PDAF (Probabilistic Data Association Filter) update.
    """
    if not meas_list or sum(betas) < 1e-6:
        return

    # Combined innovation
    H_track = H
    S_list = [H_track @ track.P @ H_track.T + m.cov for m in meas_list]

    # Use the first measurement's S for the Kalman gain (PDAF approximation)
    S = S_list[0]
    K = track.P @ H_track.T @ np.linalg.inv(S)

    # Combined innovation
    innovations = [m.pos - H_track @ track.x for m in meas_list]
    combined_innov = sum(b * v for b, v in zip(betas, innovations))

    # State update
    track.x = track.x + K @ combined_innov

    # Covariance update — PDAF formula
    Pc = (np.eye(4) - K @ H_track) @ track.P
    spread = np.zeros((4, 4))
    for b, v in zip(betas, innovations):
        spread += b * np.outer(K @ v, K @ v)
    spread -= np.outer(K @ combined_innov, K @ combined_innov)

    beta0 = max(0.0, 1.0 - sum(betas))
    track.P = beta0 * track.P + (1 - beta0) * Pc + spread

    track.hits += 1
    track.misses = 0
    for m in meas_list:
        if m.sensor not in track.last_sensors:
            track.last_sensors = (track.last_sensors + [m.sensor])[-4:]
        tg = m.extra.get("type_guess")
        if tg:
            track.drone_type_votes[tg] = track.drone_type_votes.get(tg, 0) + 1


def mark_missed(track: Track) -> None:
    track.misses += 1


def init_from_measurement(meas: Measurement) -> Track:
    """Spawn a new track from a single measurement."""
    t = Track()
    if meas.velocity is not None:
        t.x = np.array([meas.pos[0], meas.pos[1], meas.velocity[0], meas.velocity[1]])
    else:
        t.x = np.array([meas.pos[0], meas.pos[1], 0.0, 0.0])
    # Position uncertainty from sensor, velocity uncertainty large at init
    t.P = np.block([
        [meas.cov,        np.zeros((2, 2))],
        [np.zeros((2, 2)), np.eye(2) * 100.0],
    ])
    t.hits = 1
    if meas.sensor:
        t.last_sensors = [meas.sensor]
    tg = meas.extra.get("type_guess")
    if tg:
        t.drone_type_votes[tg] = 1
    return t
