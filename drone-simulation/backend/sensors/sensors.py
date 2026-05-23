"""
Sensor models.

Each sensor takes the ground-truth world and produces noisy, incomplete
measurements — the only thing the tracker ever sees.

A Measurement is a generic dict-like object: it has a position estimate (or bearing),
a measurement covariance, the sensor that produced it, and metadata used for fusion.

Jamming = sensor is degraded or silenced.
Spoofing = sensor returns fake measurements that LOOK real.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from ..sim.world import World, BASE_POS


@dataclass
class Measurement:
    sensor: str                     # 'radar' | 'camera' | 'rf' | 'acoustic'
    pos: np.ndarray                 # [x, y] — best position estimate
    cov: np.ndarray                 # 2x2 covariance matrix (uncertainty)
    bearing_only: bool = False      # True for camera/acoustic
    bearing: Optional[float] = None # radians, if bearing-only
    velocity: Optional[np.ndarray] = None  # [vx, vy] if available (radar Doppler)
    confidence: float = 1.0         # 0..1
    extra: dict = field(default_factory=dict)  # rf band, drone type guess, etc.
    spoofed: bool = False           # true if this is a fake measurement (for debug overlay)


# -------------------------------------------------------------------------
# Sensor configuration. Each sensor is positioned at a fixed location
# (could be co-located with the base, or distributed).
# -------------------------------------------------------------------------
SENSOR_POSITIONS = {
    "radar":    np.array([0.0, 0.0]),
    "camera":   np.array([0.0, 0.0]),
    "rf":       np.array([0.0, 0.0]),
    "acoustic": np.array([0.0, 0.0]),
}


class SensorSuite:
    """Wraps all four sensor models and applies jamming/spoofing."""

    def __init__(self):
        # Jamming flags — when True the sensor produces no real measurements
        # (or noisy / degraded ones depending on the sensor).
        self.jam_radar = False
        self.jam_rf = False
        self.spoof_camera = False  # camera spoofing injects ghost tracks

    # ---- RADAR -------------------------------------------------------------
    def _radar(self, world: World) -> List[Measurement]:
        """
        Radar gives (x, y) with Gaussian noise + radial velocity from Doppler.
        Range: ~3000m (realistic counter-UAS surveillance radar).
        Miss rate: ~10%. RCS-dependent (FPV drones miss more often).
        When jammed: 60% miss rate and noise variance triples.
        """
        out: List[Measurement] = []
        max_range = 3000.0
        sensor_pos = SENSOR_POSITIONS["radar"]

        base_miss = 0.10
        noise_std = 10.0
        if self.jam_radar:
            base_miss = 0.60
            noise_std = 35.0

        for d in world.drones:
            if not d.alive:
                continue
            r = np.linalg.norm(d.pos - sensor_pos)
            if r > max_range:
                continue

            # FPV drones (tiny RCS) miss more often; also degrade at long range
            rcs_penalty = 0.15 if d.drone_type == "fpv" else 0.0
            range_penalty = 0.2 * (r / max_range) ** 2  # extra miss at edge of range
            miss_p = min(0.85, base_miss + rcs_penalty + range_penalty)
            if np.random.random() < miss_p:
                continue

            noise = np.random.normal(0, noise_std, size=2)
            measured_pos = d.pos + noise
            cov = np.eye(2) * (noise_std ** 2)

            # Radial velocity (Doppler)
            radial_unit = (d.pos - sensor_pos) / r if r > 1e-3 else np.zeros(2)
            radial_speed = float(np.dot(d.vel, radial_unit))
            measured_vel = radial_unit * radial_speed + np.random.normal(0, 0.5, size=2)

            out.append(Measurement(
                sensor="radar",
                pos=measured_pos,
                cov=cov,
                velocity=measured_vel,
                confidence=0.85 if not self.jam_radar else 0.4,
            ))

        if self.jam_radar:
            n_false = np.random.poisson(2.0)
            for _ in range(n_false):
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(200, max_range)
                fake_pos = sensor_pos + r * np.array([np.cos(angle), np.sin(angle)])
                out.append(Measurement(
                    sensor="radar",
                    pos=fake_pos,
                    cov=np.eye(2) * (noise_std ** 2),
                    confidence=0.2,
                    spoofed=True,
                ))
        return out

    # ---- CAMERA (EO/IR) ----------------------------------------------------
    def _camera(self, world: World) -> List[Measurement]:
        """
        Camera gives a bearing + a rough range estimate from apparent size.
        Field of view: 60° centred north. Range: ~600m.
        Returns position with high cross-range uncertainty (ellipse along LOS).
        Spoofing: injects ghost drones in the FOV.
        """
        out: List[Measurement] = []
        sensor_pos = SENSOR_POSITIONS["camera"]
        fov_centre = np.pi / 2     # pointing north (90°)
        fov_half = np.radians(30)
        max_range = 600.0

        for d in world.drones:
            if not d.alive:
                continue
            rel = d.pos - sensor_pos
            r = np.linalg.norm(rel)
            if r > max_range or r < 10:
                continue
            bearing = np.arctan2(rel[1], rel[0])
            db = ((bearing - fov_centre + np.pi) % (2 * np.pi)) - np.pi
            if abs(db) > fov_half:
                continue
            miss_p = 0.05 + 0.3 * (r / max_range)
            if np.random.random() < miss_p:
                continue

            bearing_noisy = bearing + np.random.normal(0, np.radians(0.8))
            range_noisy = r * np.random.uniform(0.75, 1.25)
            measured_pos = sensor_pos + range_noisy * np.array([
                np.cos(bearing_noisy), np.sin(bearing_noisy)
            ])
            cov = self._directional_cov(bearing_noisy,
                                        cross_std=range_noisy * np.radians(1.0),
                                        along_std=range_noisy * 0.2)

            out.append(Measurement(
                sensor="camera",
                pos=measured_pos,
                cov=cov,
                bearing_only=False,
                bearing=float(bearing_noisy),
                confidence=0.75,
                extra={"type_guess": d.drone_type},
            ))

        if self.spoof_camera:
            for _ in range(np.random.randint(3, 6)):
                b = fov_centre + np.random.uniform(-fov_half, fov_half)
                r = np.random.uniform(150, max_range)
                pos = sensor_pos + r * np.array([np.cos(b), np.sin(b)])
                cov = self._directional_cov(b,
                                            cross_std=r * np.radians(1.0),
                                            along_std=r * 0.2)
                out.append(Measurement(
                    sensor="camera", pos=pos, cov=cov,
                    bearing_only=False, bearing=float(b),
                    confidence=0.7,
                    extra={"type_guess": "quad"},
                    spoofed=True,
                ))
        return out

    # ---- RF spectrum monitor ----------------------------------------------
    def _rf(self, world: World) -> List[Measurement]:
        """
        RF passively listens for control links.

        KEY DESIGN DECISION:
        RF gives a good bearing (±3°) but the range estimate from signal strength
        is inherently poor. Rather than using a fantasy huge along-LOS ellipse that
        destroys the Mahalanobis gate, we model it honestly:

          - cross-LOS std: r * tan(3°) ≈ r * 0.052  (bearing uncertainty)
          - along-LOS std:  HARD CAPPED at 300 m regardless of range
            (we commit to a rough range bin rather than pretending we have no range info)

        This keeps the measurement covariance usable by the JPDA gate while still
        being much larger than a radar measurement.

        Cannot detect drones with no RF link.
        """
        out: List[Measurement] = []
        sensor_pos = SENSOR_POSITIONS["rf"]
        max_range = 3000.0

        BEARING_STD_RAD = np.radians(3.0)   # ±3° bearing accuracy
        ALONG_STD_MAX   = 300.0              # hard cap: never pretend range is unknowable

        if self.jam_rf:
            n_noise = np.random.poisson(3.0)
            for _ in range(n_noise):
                bearing = np.random.uniform(-np.pi, np.pi)
                r = np.random.uniform(500, max_range)
                pos = sensor_pos + r * np.array([np.cos(bearing), np.sin(bearing)])
                # Jammer noise: use a large but still bounded covariance
                cov = self._directional_cov(bearing,
                                            cross_std=r * BEARING_STD_RAD,
                                            along_std=min(ALONG_STD_MAX * 1.5, r * 0.4))
                out.append(Measurement(
                    sensor="rf", pos=pos, cov=cov,
                    bearing_only=True, bearing=float(bearing),
                    confidence=0.15,
                    extra={"band": "wideband_noise"},
                    spoofed=True,
                ))
            return out

        for d in world.drones:
            if not d.alive or not d.rf_active:
                continue
            rel = d.pos - sensor_pos
            r = np.linalg.norm(rel)
            if r > max_range:
                continue
            if np.random.random() < 0.15:
                continue

            bearing = np.arctan2(rel[1], rel[0])
            bearing_noisy = bearing + np.random.normal(0, BEARING_STD_RAD)

            # Range estimate: log-scale signal-strength model + 30% error, hard capped
            # The estimate is genuinely useful (not random) — roughly within ±30% of truth.
            range_guess = r * np.random.uniform(0.75, 1.30)
            cross_std = range_guess * np.tan(BEARING_STD_RAD * 2)
            along_std = min(ALONG_STD_MAX, range_guess * 0.25)

            pos = sensor_pos + range_guess * np.array([
                np.cos(bearing_noisy), np.sin(bearing_noisy)
            ])
            cov = self._directional_cov(bearing_noisy, cross_std=cross_std, along_std=along_std)

            band = {"quad": "2.4GHz", "fpv": "5.8GHz", "fixed_wing": "915MHz"}[d.drone_type]
            out.append(Measurement(
                sensor="rf", pos=pos, cov=cov,
                bearing_only=True, bearing=float(bearing_noisy),
                confidence=0.60,
                extra={"band": band},
            ))
        return out

    # ---- Acoustic ----------------------------------------------------------
    def _acoustic(self, world: World) -> List[Measurement]:
        """
        Microphone array: bearing + very rough range. Short range only (~150m).
        Effective only when drones are close.
        """
        out: List[Measurement] = []
        sensor_pos = SENSOR_POSITIONS["acoustic"]
        max_range = 150.0
        for d in world.drones:
            if not d.alive:
                continue
            rel = d.pos - sensor_pos
            r = np.linalg.norm(rel)
            if r > max_range:
                continue
            if np.random.random() < 0.25:
                continue
            bearing = np.arctan2(rel[1], rel[0])
            bearing_noisy = bearing + np.random.normal(0, np.radians(5.0))
            range_guess = r * np.random.uniform(0.6, 1.5)
            pos = sensor_pos + range_guess * np.array([
                np.cos(bearing_noisy), np.sin(bearing_noisy)
            ])
            cov = self._directional_cov(bearing_noisy,
                                        cross_std=range_guess * np.radians(6.0),
                                        along_std=range_guess * 0.4)
            out.append(Measurement(
                sensor="acoustic", pos=pos, cov=cov,
                bearing_only=True, bearing=float(bearing_noisy),
                confidence=0.6,
            ))
        return out

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _directional_cov(bearing: float, cross_std: float, along_std: float) -> np.ndarray:
        """Build a 2x2 covariance ellipse aligned with the line-of-sight."""
        # Rotate diag(along^2, cross^2) by bearing
        c, s = np.cos(bearing), np.sin(bearing)
        R = np.array([[c, -s], [s, c]])
        D = np.diag([along_std ** 2, cross_std ** 2])
        return R @ D @ R.T

    # ---- public ------------------------------------------------------------
    def sense(self, world: World) -> List[Measurement]:
        meas = []
        meas.extend(self._radar(world))
        meas.extend(self._camera(world))
        meas.extend(self._rf(world))
        meas.extend(self._acoustic(world))
        return meas
