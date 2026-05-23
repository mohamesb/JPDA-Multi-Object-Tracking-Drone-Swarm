# Stendr-Style Counter-Drone Swarm Simulation

A complete simulation environment for testing multi-sensor fusion and multi-target tracking
against simulated drone swarms — built for macOS, runs entirely in software.

The frontend uses **Leaflet + OpenStreetMap (CARTO dark)** centered on **Kongsberg, Norway**
(the home of Kongsberg Defence & Aerospace — a real-world counter-UAS context).

## What this is

- **Simulates a drone swarm** flying at realistic urban-surveillance speeds (4-20 m/s)
- **Simulates 4 sensor types**: Radar, EO/IR Camera, RF spectrum monitor, Acoustic array
- **Two tracking modes**: Single-target Kalman Filter (per track, naive association) and
  Multi-target JPDA (Joint Probabilistic Data Association)
- **Jamming & spoofing controls**: toggle from the UI to see how the system degrades
- **Real city map** with dark cartography overlay, range rings, camera FOV cone
- **Dark-mode tactical UI**: click any track on the map to inspect its fused state in a
  side drawer (kinematics, sensor contributions, threat score, prediction)

## Architecture

```
backend/
  sim/world.py            # Swarm dynamics, scenarios (slow urban-realistic speeds)
  sensors/                # Radar, Camera, RF, Acoustic + jamming/spoofing
  tracking/
    kalman.py             # Constant-velocity Kalman filter
    jpda.py               # JPDA multi-target association
    fusion.py             # Threat scoring + per-sensor confidence
  server.py               # FastAPI + WebSocket server
frontend/
  index.html              # Tactical display (Leaflet map)
  app.js                  # Map rendering, controls, detail drawer
  styles.css              # Dark UI styling + Leaflet overrides
```

## Setup (macOS)

```bash
cd drone-sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m backend.server
```

Then open http://localhost:8000 in your browser.

The frontend needs an internet connection on first load to fetch the map tiles
and Leaflet itself from CDN.

## Controls

- **Launch Swarm** — spawn a coordinated pincer attack (12 drones, two arcs)
- **Line Formation** — spawn a 10-drone line approach
- **Pause / Resume** — freeze the simulation
- **Tracker** — switch between Kalman (naive) and JPDA
- **Jam Radar / Jam RF / Spoof Camera** — inject sensor degradation
- **Click any track** on the map to see its fused state in the side drawer

## Changing the deployed location

In `backend/server.py`, change `base_lat` and `base_lon` in `build_payload()` to
relocate the simulated base. Example coordinates:
- Kongsberg, NO: `59.6675, 9.6504`
- Oslo, NO:      `59.9139, 10.7522`
- Bardufoss, NO: `69.0566, 18.5403`
