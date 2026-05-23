"""
Threat scoring / fusion layer.

Takes confirmed tracks and produces a prioritized threat list:
  - Time to base (TTB)
  - Course toward base (closing velocity)
  - Speed
  - Sensor confidence (more sensors confirming = higher confidence)
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict
from .kalman import Track
from ..sim.world import BASE_POS


def threat_score(track: Track) -> Dict[str, float]:
    pos = track.pos
    vel = track.vel
    r = float(np.linalg.norm(pos - BASE_POS))
    speed = float(np.linalg.norm(vel))

    # Closing velocity (positive = closing toward base)
    if r > 1e-3:
        to_base = (BASE_POS - pos) / r
        closing = float(np.dot(vel, to_base))
    else:
        closing = 0.0

    # Time to base (only meaningful if closing)
    if closing > 1.0:
        ttb = r / closing
    else:
        ttb = float("inf")

    # Sensor confidence — multi-sensor agreement raises score
    n_sensors = len(set(track.last_sensors))
    sensor_factor = min(1.0, 0.4 + 0.2 * n_sensors)

    # Raw score: higher = more dangerous
    # Closer + faster closing + multi-sensor = more dangerous.
    # Speed ranges are now urban-realistic (4-20 m/s), so we normalise to 15 m/s.
    proximity = max(0.0, 1.0 - r / 2500.0)
    closing_norm = max(0.0, min(1.0, closing / 15.0))
    raw = (0.55 * proximity + 0.35 * closing_norm + 0.10 * (speed / 20.0)) * sensor_factor

    level = "low"
    if raw > 0.65:
        level = "critical"
    elif raw > 0.45:
        level = "high"
    elif raw > 0.25:
        level = "medium"

    drone_type = None
    if track.drone_type_votes:
        drone_type = max(track.drone_type_votes.items(), key=lambda kv: kv[1])[0]

    return {
        "range_m": r,
        "speed_mps": speed,
        "closing_mps": closing,
        "ttb_s": ttb if ttb != float("inf") else None,
        "score": float(raw),
        "level": level,
        "n_sensors": n_sensors,
        "drone_type": drone_type,
    }
