"""
Two multi-target trackers built on top of the Kalman filter:

  1. NaiveTracker (single-target Kalman, naive association)
     - For each track, picks the single nearest measurement (gating) and updates with it.
     - Fast and simple. Breaks down when tracks cross or measurements are dense.

  2. JPDATracker (Joint Probabilistic Data Association)
     - For each track, looks at ALL measurements inside a validation gate.
     - Computes association probabilities (betas) for each measurement.
     - Updates the track with a weighted combination of all candidate measurements.
     - Robust to crossing tracks, ambiguous detections, and clutter from jamming.

Both share the same external API:
    tracker.step(measurements, dt) -> List[Track]
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Set
from scipy.optimize import linear_sum_assignment

from . import kalman
from .kalman import Track, predict, update, update_weighted, mark_missed, init_from_measurement, H
from ..sensors.sensors import Measurement


# ---- Tunable parameters ---------------------------------------------------
GATE_CHI2 = 9.21          # 99% gate for 2-DOF chi-square (Mahalanobis squared)
CONFIRM_HITS = 5          # promotions to confirmed after this many hits
CONFIRM_HITS_RF_ONLY = 12 # RF-bearing-only tracks need more evidence before confirming
DELETE_MISSES = 20        # drop a track after this many consecutive misses
INIT_DISTANCE = 50.0      # fallback Euclidean guard (meters) — mainly a safety net
P_DETECTION = 0.85        # assumed probability of detection (per sensor visit)
CLUTTER_DENSITY = 1e-6    # expected clutter measurements per m^2


# ===========================================================================
# Shared helpers
# ===========================================================================
def _mahalanobis_sq(track: Track, meas: Measurement) -> float:
    """Squared Mahalanobis distance between a predicted track and a measurement."""
    y = meas.pos - H @ track.x
    S = H @ track.P @ H.T + meas.cov
    try:
        return float(y @ np.linalg.inv(S) @ y)
    except np.linalg.LinAlgError:
        return float("inf")


def _gaussian_likelihood(track: Track, meas: Measurement) -> float:
    """N(z; Hx, S) — used in JPDA to weight associations."""
    y = meas.pos - H @ track.x
    S = H @ track.P @ H.T + meas.cov
    try:
        Sinv = np.linalg.inv(S)
        det = np.linalg.det(S)
        if det <= 0:
            return 0.0
        norm = 1.0 / (2 * np.pi * np.sqrt(det))
        return float(norm * np.exp(-0.5 * y @ Sinv @ y))
    except np.linalg.LinAlgError:
        return 0.0


def _spawn_from_unused(
    measurements: List[Measurement],
    unused_indices: Set[int],
    existing_tracks: List[Track],
) -> List[Track]:
    """
    Spawn new tracks from measurements that no existing track gates.

    Two-pass approach to avoid ghost duplicates:
      1. Filter out measurements covered by any existing track (Mahalanobis gate).
      2. Greedily cluster the remaining measurements by Euclidean proximity,
         spawning at most ONE track per cluster (from the most informative measurement).

    This prevents e.g. two adjacent radar pings of the same drone each spawning their
    own track, which happened when both were spawned simultaneously before either had
    updated its P enough to gate the other.
    """
    # --- Pass 1: remove measurements already gated by existing tracks ------
    candidates = []
    for j in unused_indices:
        m = measurements[j]
        if any(_mahalanobis_sq(t, m) < GATE_CHI2 * 2.0 for t in existing_tracks):
            continue  # covered
        candidates.append(m)

    # --- Pass 2: cluster remaining measurements ----------------------------
    CLUSTER_DIST = 200.0   # meters — measurements closer than this share a cluster
    used = set()
    new_tracks = []

    for i, m in enumerate(candidates):
        if i in used:
            continue
        # Build cluster
        cluster = [i]
        used.add(i)
        for j, m2 in enumerate(candidates):
            if j in used:
                continue
            if np.linalg.norm(m.pos - m2.pos) < CLUSTER_DIST:
                cluster.append(j)
                used.add(j)

        # Pick the most informative measurement to initialise from.
        # Prefer radar (smallest covariance trace) > camera > rf > acoustic.
        SENSOR_PRIO = {"radar": 0, "camera": 1, "rf": 2, "acoustic": 3}
        best = min(cluster,
                   key=lambda k: (SENSOR_PRIO.get(candidates[k].sensor, 9),
                                  np.trace(candidates[k].cov)))
        new_tracks.append(init_from_measurement(candidates[best]))

    return new_tracks


def _merge_duplicate_tracks(tracks: List[Track]) -> List[Track]:
    """
    Merge confirmed tracks that are statistically indistinguishable.

    Two tracks are merged when their positions are within the combined 1-sigma
    ellipses AND both are confirmed. We keep the older (more hits) track and
    drop the younger one, blending their state with an inverse-covariance average.

    This is the final safeguard against JPDA's 'probability splitting' problem:
    when a single measurement is gated by N near-parallel tracks, each gets a
    beta of ~1/N and all update from the same drone. The tracks converge to the
    same position but are never dropped. The merger step catches them.
    """
    # Only merge confirmed tracks — tentative ones will time out naturally.
    confirmed = [t for t in tracks if t.confirmed]
    unconfirmed = [t for t in tracks if not t.confirmed]

    if len(confirmed) < 2:
        return tracks

    MERGE_DIST_M = 120.0  # merge if track positions are within this many meters
    merged_ids: Set[int] = set()
    out_confirmed: List[Track] = []

    for i, a in enumerate(confirmed):
        if a.id in merged_ids:
            continue
        absorbed = []
        for j, b in enumerate(confirmed):
            if i >= j or b.id in merged_ids:
                continue
            if np.linalg.norm(a.pos - b.pos) < MERGE_DIST_M:
                absorbed.append(b)

        if not absorbed:
            out_confirmed.append(a)
            continue

        # Keep the track with more hits; blend its state with each absorbed track
        # using information-form averaging (weight by inverse-covariance).
        keeper = max([a] + absorbed, key=lambda t: t.hits)
        others = [t for t in [a] + absorbed if t is not keeper]

        # Blend: x_merged = P_merged @ sum_i(P_i^-1 @ x_i)
        try:
            info_sum = np.linalg.inv(keeper.P)
            x_sum = info_sum @ keeper.x
            for o in others:
                Pinv = np.linalg.inv(o.P)
                info_sum += Pinv
                x_sum += Pinv @ o.x
            keeper.P = np.linalg.inv(info_sum)
            keeper.x = keeper.P @ x_sum
            keeper.hits = max(keeper.hits, max((o.hits for o in others), default=0))
        except np.linalg.LinAlgError:
            pass  # degenerate covariance — keep keeper unchanged

        for o in others:
            merged_ids.add(o.id)
        out_confirmed.append(keeper)

    return out_confirmed + unconfirmed


def _maintenance(tracks: List[Track]) -> List[Track]:
    """Confirm new tracks, drop stale ones."""
    out = []
    for t in tracks:
        position_sensors = set(t.last_sensors) - {"rf", "acoustic"}
        thresh = CONFIRM_HITS if position_sensors else CONFIRM_HITS_RF_ONLY
        if t.hits >= thresh:
            t.confirmed = True
        if t.misses >= DELETE_MISSES:
            continue
        out.append(t)
    return out


# ===========================================================================
# Naive single-target Kalman tracker
# ===========================================================================
def _maintenance(tracks: List[Track]) -> List[Track]:
    """Confirm new tracks, drop stale ones."""
    out = []
    for t in tracks:
        position_sensors = set(t.last_sensors) - {"rf", "acoustic"}
        thresh = CONFIRM_HITS if position_sensors else CONFIRM_HITS_RF_ONLY
        if t.hits >= thresh:
            t.confirmed = True
        if t.misses >= DELETE_MISSES:
            continue
        out.append(t)
    return out


class NaiveTracker:
    """
    For each track, pick the single best measurement (lowest Mahalanobis distance)
    inside the gate. Greedy, fast, fragile.
    """
    name = "kalman"

    def __init__(self):
        self.tracks: List[Track] = []

    def step(self, measurements: List[Measurement], dt: float) -> List[Track]:
        # 1. Predict all tracks forward
        for t in self.tracks:
            predict(t, dt)

        used_meas: Set[int] = set()

        # 2. Greedy assignment via Hungarian — but each track gets at most one meas
        if self.tracks and measurements:
            cost = np.full((len(self.tracks), len(measurements)), 1e6)
            for i, t in enumerate(self.tracks):
                for j, m in enumerate(measurements):
                    d2 = _mahalanobis_sq(t, m)
                    if d2 < GATE_CHI2:
                        cost[i, j] = d2
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < GATE_CHI2:
                    update(self.tracks[i], measurements[j])
                    used_meas.add(j)
                else:
                    mark_missed(self.tracks[i])
            # Tracks not in row_ind got nothing
            assigned_tracks = set(row_ind.tolist())
            for i, t in enumerate(self.tracks):
                if i not in assigned_tracks:
                    mark_missed(t)
        else:
            for t in self.tracks:
                mark_missed(t)

        # 3. Spawn new tracks with clustering to prevent ghost duplicates.
        self.tracks.extend(_spawn_from_unused(measurements, set(range(len(measurements))) - used_meas, self.tracks))
        self.tracks = _merge_duplicate_tracks(self.tracks)

        # 4. Maintenance
        self.tracks = _maintenance(self.tracks)
        return self.tracks


# ===========================================================================
# JPDA tracker
# ===========================================================================
class JPDATracker:
    """
    Joint Probabilistic Data Association.

    For each track, compute association probabilities to ALL measurements in
    its gate (plus a "no detection" hypothesis). Apply a weighted update.

    This implementation uses the simplified JPDA where we compute per-track
    betas via likelihood weighting — sufficient to be visibly better than naive
    in swarm scenarios, and dramatically simpler than full joint hypothesis
    enumeration.
    """
    name = "jpda"

    def __init__(self):
        self.tracks: List[Track] = []

    def step(self, measurements: List[Measurement], dt: float) -> List[Track]:
        # 1. Predict
        for t in self.tracks:
            predict(t, dt)

        # 2. Build gated likelihoods: L[i][j] = likelihood that meas j came from track i
        n_tracks = len(self.tracks)
        n_meas = len(measurements)
        if n_tracks == 0:
            self.tracks.extend(_spawn_from_unused(measurements, set(range(len(measurements))), []))
            self.tracks = _maintenance(self.tracks)
            return self.tracks

        L = np.zeros((n_tracks, n_meas))
        gated = [[False] * n_meas for _ in range(n_tracks)]
        for i, t in enumerate(self.tracks):
            for j, m in enumerate(measurements):
                d2 = _mahalanobis_sq(t, m)
                if d2 < GATE_CHI2:
                    L[i, j] = _gaussian_likelihood(t, m)
                    gated[i][j] = True

        # 3. For each track, compute beta_ij (association probabilities).
        #    Simplified marginal JPDA: normalise likelihoods across gated measurements
        #    plus a "no-detection" hypothesis. The no-detect weight represents the
        #    probability that all gated measurements are clutter and the true target
        #    was not detected: (1 - P_D) * lambda^m  where lambda is clutter density.
        all_used_meas: Set[int] = set()
        for i, t in enumerate(self.tracks):
            # Likelihood-weighted by detection probability
            row = L[i] * P_DETECTION
            n_gated = int(sum(gated[i]))
            if n_gated == 0:
                mark_missed(t)
                continue

            # No-detection hypothesis weight. Use a small constant so a gated meas
            # with even modest likelihood dominates, rather than the no-detect option.
            no_detect_weight = (1.0 - P_DETECTION) * CLUTTER_DENSITY

            denom = row.sum() + no_detect_weight
            if denom < 1e-20:
                mark_missed(t)
                continue
            betas = row / denom

            # Build the list of measurements that passed the gate
            cand_meas = [measurements[j] for j in range(n_meas) if gated[i][j]]
            cand_betas = [float(betas[j]) for j in range(n_meas) if gated[i][j]]

            total_assoc = sum(cand_betas)
            if not cand_meas or total_assoc < 1e-6:
                mark_missed(t)
                continue

            update_weighted(t, cand_meas, cand_betas)
            for j in range(n_meas):
                if gated[i][j] and betas[j] > 0.1:
                    all_used_meas.add(j)

        # 4. Spawn new tracks with clustering to prevent ghost duplicates.
        self.tracks.extend(_spawn_from_unused(measurements, set(range(n_meas)) - all_used_meas, self.tracks))
        self.tracks = _merge_duplicate_tracks(self.tracks)

        # 5. Maintenance
        self.tracks = _maintenance(self.tracks)
        return self.tracks
