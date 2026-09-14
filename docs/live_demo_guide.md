# NAVIGUARD Live Demonstration & Operator Guide

This guide provides simple, step-by-step instructions to launch, observe, and interact with the complete **NAVIGUARD** autonomous outdoor UGV system (Phases 1 through 8).

> [!NOTE]
> **Demo Motion vs. Autonomous Mission Navigation**:
> The motion generated during this live demonstration is a **deterministic visualization trajectory** (`naviguard_demo_motion`). It safely moves the robot in bounded smooth loops ($v_x \le 0.35$ m/s, $|\omega_z| \le 0.35$ rad/s) to excite cameras, IMU, VO, and SLAM pipelines.
> **Phase 9 (global A*/Dijkstra path planning, Nav2, obstacle avoidance)** is deliberately not included in this demonstration foundation.

---

## 1. Quick Start (Single Command)

From the workspace root (`~/naviguard_ws`), launch the complete simulation and software stack:

```bash
# In Terminal 1: Launch full simulation + complete 8-phase pipeline
./scripts/start_naviguard_demo.sh
```

### Launch Options

| Command Flag | Description |
| :--- | :--- |
| *(default)* | Runs Gazebo Harmonic simulation with GUI, starts all nodes + demo motion |
| `--headless` | Runs Gazebo simulation headless (server-only, ideal for WSL2 without GUI/X11) |
| `--with-rviz` | Automatically starts RViz2 with the complete NAVIGUARD demo profile |
| `--no-motion` | Robot spawns stationary (no automatic demo motion commanded) |
| `--rviz-only` | Opens RViz2 standalone to attach to an already running simulation |

Example:
```bash
# Running in WSL2 / headless terminal:
./scripts/start_naviguard_demo.sh --headless
```

---

## 2. Real-Time System Monitoring

Open a second terminal to check the health and state transitions of the entire system at any time:

### Live Dashboard: `naviguard_status.sh`
```bash
./scripts/naviguard_status.sh
```

**Sample Dashboard Output**:
```text
====================================================================
           NAVIGUARD LIVE SYSTEM STATUS DASHBOARD
====================================================================

[1] PIPELINE NODES:
  Node Name                      | Status    
  ------------------------------------------
  robot_state_publisher          | ONLINE    
  ros_gz_bridge                  | ONLINE    
  naviguard_perception_node      | ONLINE    
  visual_odometry_node           | ONLINE    
  sensor_sync_node               | ONLINE    
  state_estimation_node          | ONLINE    
  slam_node                      | ONLINE    
  confidence_node                | ONLINE    
  recovery_node                  | ONLINE    
  naviguard_demo_motion          | ONLINE    
  >> 10/10 Core Nodes Active

[2] SUBSYSTEM STATES:
  Confidence Decision : CONTINUE (Composite Score: 0.92, Mode: NOMINAL)
  Recovery State      : NORMAL (Strategy: NONE, Dwell: 14.2s)
  Demo Motion         : ACTIVE [Mode: LOOP_FORWARD (3.2/6.0s), cmd_vx=0.25, cmd_wz=0.00]

[3] POSE & ODOMETRY ESTIMATION:
  State Estimation    : pos=(2.14, 0.48) m, vx=0.25 m/s
  SLAM Global Pose    : x=2.12, y=0.47, z=0.00 m
====================================================================
```

### Topic Inventory & Active Status: `naviguard_topics.sh`
```bash
./scripts/naviguard_topics.sh
```
Displays all 25 active topics across simulation, perception, VO, sync, state estimation, SLAM, confidence, and recovery.

---

## 3. Visualizing Sensor & Pipeline Streams

Inspect live camera views and algorithm debug overlays using the provided viewing scripts:

```bash
# View raw monocular camera feed:
./scripts/view_raw_camera.sh

# View perception feature overlay (edges, sky/ground segmentation):
./scripts/view_perception.sh

# View visual odometry feature tracking (Shi-Tomasi + Lucas-Kanade optical flow):
./scripts/view_visual_odometry.sh
```

> [!TIP]
> In environments with an X11 / Wayland display (WSLg), these scripts automatically open interactive OpenCV preview windows.
> In headless mode or with the `--snapshot` flag, they capture the active frame and save high-resolution PNG snapshots directly to `/tmp/`.

---

## 4. RViz2 Integrated Demonstration Layout

To view the robot, SLAM map, trajectory, landmarks, confidence halo, and recovery checkpoints in 3D:

```bash
./scripts/start_naviguard_demo.sh --rviz-only
```

Pre-configured displays in `naviguard_demo.rviz`:
* **RobotModel**: 4-wheel outdoor differential-drive UGV (`/robot_description`)
* **TF Tree**: `map` $\to$ `odom` $\to$ `base_footprint` $\to$ `base_link` $\to$ `camera_link`
* **SLAM Map**: 2D occupancy grid (`/slam/map`)
* **SLAM Trajectory**: Red path showing estimated historical trajectory (`/slam/trajectory`)
* **SLAM Robot Pose**: Green arrow depicting real-time pose estimate (`/slam/pose`)
* **SLAM Landmarks**: Visual 3D feature landmarks (`/slam/landmarks`)
* **Confidence Decision**: Dynamic visual status ring (`/naviguard/decision_marker`)
* **Recovery Visualization**: Safe backtracking checkpoints, safe targets, and path history (`/recovery/visualization`)
* **Live Camera & VO Stream**: Embedded real-time imagery displays

---

## 5. Controlled Fault Injection & Autonomous Recovery

Test NAVIGUARD's closed-loop resilience by injecting controlled faults:

```bash
# Interactive mode (menu selection):
./scripts/inject_fault.sh

# Or trigger manual recovery directly:
./scripts/inject_fault.sh trigger

# Reset recovery budget and return to NORMAL operation:
./scripts/inject_fault.sh reset

# Inspect current recovery FSM state:
./scripts/inject_fault.sh status
```

### What Happens During Recovery:
1. **Fault Triggered**: Recovery FSM transitions: `NORMAL` $\to$ `SAFE_STOP` $\to$ `RECOVER`.
2. **Mutual Exclusion**: `naviguard_demo_motion` instantly detects the recovery state, halts velocity commands, and yields `/cmd_vel`.
3. **Autonomous Action**: `recovery_node` takes command of `/cmd_vel` to execute a controlled `SHORT_BACKTRACK` to a high-confidence trusted checkpoint.
4. **Relocalization & Resume**: After reaching the trusted checkpoint, the robot verifies sensor convergence and transitions through `RELOCALIZE` $\to$ `RESUME` $\to$ `NORMAL`. Demo motion gently resumes.
