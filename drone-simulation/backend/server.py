"""
FastAPI server.

  - Serves the static frontend at /
  - Exposes WebSocket /ws that pushes a state snapshot ~10x per second
  - Exposes HTTP POST endpoints for control buttons (launch, pause, tracker mode,
    jamming, spoofing, clear)
"""
from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .sim.world import World, DT, BASE_POS, WORLD_RADIUS
from .sensors.sensors import SensorSuite
from .tracking.jpda import NaiveTracker, JPDATracker
from .tracking.fusion import threat_score


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ---- Simulation state (single instance, single process) ------------------
class SimState:
    def __init__(self):
        self.world = World()
        self.sensors = SensorSuite()
        self.tracker_kalman = NaiveTracker()
        self.tracker_jpda = JPDATracker()
        self.active_mode = "jpda"   # 'kalman' or 'jpda'
        self.latest_measurements: list = []
        self.latency_ms = 0.0
        self.sim_speed = 3.0        # how many sim seconds per wall second (default 3x)

    def active_tracker(self):
        return self.tracker_jpda if self.active_mode == "jpda" else self.tracker_kalman

    def reset_tracker(self):
        self.tracker_kalman = NaiveTracker()
        self.tracker_jpda = JPDATracker()


state = SimState()
clients: set[WebSocket] = set()


# ---- Simulation tick ------------------------------------------------------
async def tick_loop():
    """Runs forever — advances world, runs sensors, runs tracker, broadcasts state.

    The server-side DT is fixed (10 Hz sim time), but we can compress wall time:
    every wall-clock 100 ms we step the world `sim_speed` times so the user can
    watch a slow physical scenario unfold quickly on screen. The physics, sensor
    noise, and tracker behaviour all remain identical to real-time.
    """
    while True:
        t0 = time.perf_counter()
        # Step world & tracker as many times as sim_speed dictates.
        # We always emit ONE payload per wall tick so the UI rate is stable.
        steps = max(1, int(round(state.sim_speed)))
        measurements = []
        for _ in range(steps):
            state.world.step()
            measurements = state.sensors.sense(state.world)
            tracker = state.active_tracker()
            tracks = tracker.step(measurements, DT)
        state.latest_measurements = measurements
        state.latency_ms = (time.perf_counter() - t0) * 1000.0

        payload = build_payload(tracks, measurements)
        await broadcast(payload)
        await asyncio.sleep(DT)


def _match_tracks_to_truth(tracks, truth):
    """
    Hungarian-assignment: pair each track to its closest ground-truth drone.
    Returns: dict { track_id: { 'truth_id', 'pos_error_m', 'vel_error_mps', 'heading_error_deg' } }

    This is ONLY for evaluation/display — the tracker never sees the truth.
    """
    from scipy.optimize import linear_sum_assignment

    if not tracks or not truth:
        return {}

    cost = np.zeros((len(tracks), len(truth)))
    for i, t in enumerate(tracks):
        for j, d in enumerate(truth):
            cost[i, j] = np.hypot(t.pos[0] - d["x"], t.pos[1] - d["y"])

    row_ind, col_ind = linear_sum_assignment(cost)

    out = {}
    for i, j in zip(row_ind, col_ind):
        # Reject absurdly bad matches — a track has no business being paired to
        # a truth drone 500 m away. Better to call it unmatched.
        if cost[i, j] > 500.0:
            continue
        t = tracks[i]
        d = truth[j]
        pos_err = float(cost[i, j])
        vel_err = float(np.hypot(t.vel[0] - d["vx"], t.vel[1] - d["vy"]))
        # Heading wrapped to [-180, 180]
        hd = (t.heading - d["heading"] + 180) % 360 - 180
        out[t.id] = {
            "truth_id": d["id"],
            "truth_x": d["x"],
            "truth_y": d["y"],
            "pos_error_m": pos_err,
            "vel_error_mps": vel_err,
            "heading_error_deg": float(hd),
        }
    return out


def build_payload(tracks, measurements):
    truth = state.world.ground_truth_snapshot()
    # Build a confirmed-track shortlist first (we only evaluate accuracy on those)
    confirmed_tracks = [t for t in tracks if t.confirmed]
    accuracy = _match_tracks_to_truth(confirmed_tracks, truth)

    track_list = []
    for t in tracks:
        if not t.confirmed and t.hits < 2:
            continue
        ts = threat_score(t)
        acc = accuracy.get(t.id)
        track_list.append({
            "id": t.id,
            "x": float(t.pos[0]),
            "y": float(t.pos[1]),
            "vx": float(t.vel[0]),
            "vy": float(t.vel[1]),
            "speed": t.speed,
            "heading": t.heading,
            "confirmed": t.confirmed,
            "hits": t.hits,
            "misses": t.misses,
            "age": t.age,
            "sensors": t.last_sensors,
            "cov_trace": float(np.trace(t.P[:2, :2])),
            "accuracy": acc,  # may be None if unmatched
            **ts,
        })

    meas_list = []
    for m in measurements:
        meas_list.append({
            "sensor": m.sensor,
            "x": float(m.pos[0]),
            "y": float(m.pos[1]),
            "spoofed": m.spoofed,
            "bearing_only": m.bearing_only,
        })

    return {
        "t": state.world.time,
        "paused": state.world.paused,
        "mode": state.active_mode,
        "latency_ms": round(state.latency_ms, 2),
        "world_radius": WORLD_RADIUS,
        "dt": DT,                    # so the frontend can interpolate at the right rate
        "sim_speed": state.sim_speed,
        "base_lat": 59.9139,        # Oslo, Norway
        "base_lon": 10.7522,
        "jamming": {
            "radar": state.sensors.jam_radar,
            "rf": state.sensors.jam_rf,
            "camera_spoof": state.sensors.spoof_camera,
        },
        "tracks": track_list,
        "measurements": meas_list,
        "ground_truth": truth,
        "n_truth": len(truth),
        "n_tracks": len(track_list),
    }


async def broadcast(payload: dict):
    if not clients:
        return
    msg = json.dumps(payload)
    dead = []
    for ws in clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


# ---- Lifespan -------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(tick_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


# ---- HTTP control endpoints ----------------------------------------------
class LaunchReq(BaseModel):
    n: int = 12
    pattern: str = "pincer"


@app.post("/api/launch")
async def launch(req: LaunchReq):
    state.world.launch_swarm(n=req.n, pattern=req.pattern)
    return {"ok": True, "spawned": req.n}


@app.post("/api/pause")
async def pause():
    state.world.paused = not state.world.paused
    return {"paused": state.world.paused}


@app.post("/api/clear")
async def clear():
    state.world.clear()
    state.reset_tracker()
    return {"ok": True}


class ModeReq(BaseModel):
    mode: str  # 'kalman' or 'jpda'


@app.post("/api/mode")
async def set_mode(req: ModeReq):
    if req.mode not in ("kalman", "jpda"):
        return {"ok": False, "error": "invalid mode"}
    state.active_mode = req.mode
    state.reset_tracker()
    return {"ok": True, "mode": req.mode}


class JamReq(BaseModel):
    target: str  # 'radar' | 'rf' | 'camera_spoof'
    on: bool


@app.post("/api/jam")
async def jam(req: JamReq):
    if req.target == "radar":
        state.sensors.jam_radar = req.on
    elif req.target == "rf":
        state.sensors.jam_rf = req.on
    elif req.target == "camera_spoof":
        state.sensors.spoof_camera = req.on
    else:
        return {"ok": False, "error": "invalid target"}
    return {"ok": True}


class SpeedReq(BaseModel):
    speed: float


@app.post("/api/speed")
async def set_speed(req: SpeedReq):
    state.sim_speed = max(1.0, min(20.0, req.speed))
    return {"ok": True, "speed": state.sim_speed}


# ---- WebSocket -----------------------------------------------------------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            # Keep the connection open; ignore inbound messages.
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)
    except Exception:
        clients.discard(websocket)


# ---- Static frontend -----------------------------------------------------
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
