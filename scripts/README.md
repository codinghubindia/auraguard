# NAVIGUARD Operator & Demonstration Scripts Guide

This directory contains executable scripts to launch, monitor, test, and safely terminate the entire NAVIGUARD autonomous robotics stack.

---

## 1. Master Demonstration (One-Command Start)

To start the complete system (Gazebo simulation, UGV, perception, visual odometry, SLAM, confidence engine, recovery, A* global navigation, and operator dashboard):

```bash
cd ~/naviguard_ws
source /opt/ros/jazzy/setup.bash
source ~/naviguard_ws/install/setup.bash

./scripts/run_naviguard.sh
```

### Command-line Options:
- `--headless` (default): Runs Gazebo server without GUI for optimal performance in WSL2/WSLg.
- `--gui`: Launches Gazebo Harmonic with the full 3D simulation window.
- `--with-rviz`: Opens RViz2 alongside the dashboard with `naviguard_final.rviz`.
- `--no-browser`: Prevents auto-launching the web browser.

---

## 2. Operator Web Dashboard

Once started, navigate to:
**`http://localhost:8080`**

### Operator Workflow:
1. View live camera feeds by toggling **RAW**, **PERCEPTION / TRAVERSABILITY**, and **VO HUD**.
2. Click **SET DESTINATION** in the bottom control bar.
3. Click anywhere on the map (e.g. ahead along the off-road dirt trail).
4. Verify destination coordinates ($X, Y$) and safety validation badge.
5. Click **START NAVIGATION**.
6. Observe autonomous trajectory execution, obstacle replanning, and confidence transitions.

---

## 3. Clean System Shutdown

To cleanly terminate all ROS nodes, simulation processes, bridges, and background servers without leaving orphan processes:

```bash
./scripts/stop_naviguard.sh
```
*(Or press `Ctrl+C` in the terminal running `run_naviguard.sh`)*

---

## 4. Engineering & Diagnostic Utilities

| Script | Purpose |
|---|---|
| `./scripts/naviguard_status.py` | Terminal status dashboard displaying live node states, rates, and navigation telemetry. |
| `./scripts/naviguard_topics.sh` | Echoes topic list and publication rates. |
| `./scripts/inject_fault.sh` | Injects controlled recovery faults (`--trigger`, `--reset`, `--status`). |
| `./scripts/send_navigation_goal.sh <X> <Y> [Yaw]` | Dispatches goals from CLI without the browser. |
| `./scripts/view_raw_camera.sh` | Displays `/camera/image_raw` using standalone viewer. |
| `./scripts/view_perception.sh` | Displays `/perception/debug_image`. |
| `./scripts/view_visual_odometry.sh` | Displays `/visual_odometry/debug_image`. |
