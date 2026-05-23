"""
Drone swarm simulation core.

Holds the ground-truth state of all drones in the world. The trackers and sensors
should NEVER look at this directly — they only see noisy measurements from the
sensor models. This is the "real world" we are pretending exists.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List
import itertools


# Defended position is the origin (0, 0). All distances in meters.
# In the frontend, (0, 0) corresponds to a real GPS coordinate (Kongsberg base).
BASE_POS = np.array([0.0, 0.0])
WORLD_RADIUS = 2500.0   # meters — drones spawn at this radius
DT = 0.1                # simulation tick interval (seconds) — 10 Hz

# Realistic urban-surveillance drone speeds (m/s).
# Commercial quadcopters cruise at 5-10 m/s. FPV racers go faster but rarely
# fly attack runs at top speed. Fixed-wing surveillance drones cruise 15-22 m/s.
SPEED_RANGES = {
    "quad":       (4.0, 9.0),
    "fpv":        (8.0, 14.0),
    "fixed_wing": (14.0, 20.0),
}


_id_counter = itertools.count(1)


@dataclass
class Drone:
    """A single ground-truth drone."""
    id: int
    pos: np.ndarray            # [x, y] meters
    vel: np.ndarray            # [vx, vy] m/s
    drone_type: str            # 'quad' | 'fixed_wing' | 'fpv'
    rf_active: bool = True     # is its control link emitting?
    alive: bool = True
    age: float = 0.0           # seconds since spawn

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    @property
    def heading(self) -> float:
        """Heading in degrees, 0 = east, 90 = north."""
        return float(np.degrees(np.arctan2(self.vel[1], self.vel[0])))

    @property
    def distance_to_base(self) -> float:
        return float(np.linalg.norm(self.pos - BASE_POS))


class World:
    """The simulated world — ground truth only."""

    def __init__(self):
        self.drones: List[Drone] = []
        self.time: float = 0.0
        self.paused: bool = False

    # ---- spawning ----------------------------------------------------------
    def launch_swarm(self, n: int = 12, pattern: str = "pincer") -> None:
        """Spawn a coordinated swarm attack."""
        # Clear dead drones to keep state small
        self.drones = [d for d in self.drones if d.alive]

        if pattern == "pincer":
            # Two arcs converging from northwest and northeast
            for i in range(n):
                arc_left = i < n // 2
                base_angle = np.radians(135 if arc_left else 45)
                spread = np.radians(np.random.uniform(-20, 20))
                angle = base_angle + spread
                r = WORLD_RADIUS + np.random.uniform(-100, 100)
                pos = np.array([r * np.cos(angle), r * np.sin(angle)])
                to_base = -pos / np.linalg.norm(pos)
                drone_type = np.random.choice(
                    ["quad", "fpv", "fixed_wing"], p=[0.6, 0.25, 0.15]
                )
                lo, hi = SPEED_RANGES[drone_type]
                speed = np.random.uniform(lo, hi)
                vel = to_base * speed + np.random.uniform(-0.5, 0.5, size=2)
                self.drones.append(Drone(
                    id=next(_id_counter),
                    pos=pos,
                    vel=vel,
                    drone_type=drone_type,
                    rf_active=np.random.random() > 0.15,
                ))
        elif pattern == "line":
            for i in range(n):
                angle = np.radians(90 + (i - n / 2) * 4)
                pos = np.array([WORLD_RADIUS * np.cos(angle), WORLD_RADIUS * np.sin(angle)])
                to_base = -pos / np.linalg.norm(pos)
                drone_type = "quad"
                lo, hi = SPEED_RANGES[drone_type]
                speed = np.random.uniform(lo, hi)
                vel = to_base * speed
                self.drones.append(Drone(
                    id=next(_id_counter), pos=pos, vel=vel,
                    drone_type=drone_type, rf_active=True,
                ))

    def clear(self) -> None:
        self.drones = []

    # ---- physics -----------------------------------------------------------
    def step(self) -> None:
        """Advance the simulation by DT seconds."""
        if self.paused:
            return
        self.time += DT

        for d in self.drones:
            if not d.alive:
                continue
            # Small random course adjustments (turbulence / pilot input)
            d.vel += np.random.normal(0, 0.15, size=2)
            # Soft cap on speed (per type ceiling, with a global max)
            sp = np.linalg.norm(d.vel)
            type_max = SPEED_RANGES[d.drone_type][1] * 1.2
            if sp > type_max:
                d.vel *= type_max / sp
            d.pos += d.vel * DT
            d.age += DT

            # Kill drones that reach the base (impact) or fly out of world
            if d.distance_to_base < 30.0:
                d.alive = False
            elif d.distance_to_base > WORLD_RADIUS * 1.5:
                d.alive = False

    # ---- serialization -----------------------------------------------------
    def ground_truth_snapshot(self) -> List[dict]:
        """For debug / display only — never feed this to trackers."""
        return [
            {
                "id": d.id,
                "x": float(d.pos[0]),
                "y": float(d.pos[1]),
                "vx": float(d.vel[0]),
                "vy": float(d.vel[1]),
                "speed": d.speed,
                "heading": d.heading,
                "type": d.drone_type,
                "rf_active": d.rf_active,
            }
            for d in self.drones if d.alive
        ]
