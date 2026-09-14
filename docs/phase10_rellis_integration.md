# Phase 10: RELLIS-3D Integration, Realistic Environment & Offline Evaluation Report

**Project:** NAVIGUARD — Confidence-Aware Closed-Loop Navigation & Autonomous Recovery for an Outdoor UGV  
**Phase:** 10 (Final Engineering Foundation Phase)  
**Status:** COMPLETE & VERIFIED (178 / 178 Tests Passing across 10 Packages)

---

## 1. Executive Summary

Phase 10 establishes realistic outdoor benchmarking and environment representation for NAVIGUARD without compromising the camera-primary navigation autonomy:
1. **Mode A — Realistic Gazebo Harmonic Simulation Environment (`rellis_outdoor_world.sdf`)**:
   - High-fidelity outdoor world modeling RELLIS-3D unstructured off-road trail conditions.
   - Distinct surface physics: high-traction dirt trail ($\mu = 0.85$), grass terrain, low-friction mud patches ($\mu = 0.22$).
   - Authentic natural obstacles: tree trunks, dense foliage, fallen timber log barriers, and rock cairns.
   - Decoupled visual and collision geometry (REP-compliant lightweight collision primitives for real-time physics stability).
2. **Mode B — RELLIS-3D Dataset Replay & Evaluation Package (`naviguard_rellis`)**:
   - Full dataset validation, sequence ingestion, camera calibration loading, and frame replay.
   - Ground Truth Isolation Barrier (`ground_truth_guard.py`): Architectural firewall guaranteeing zero leakage of ground-truth poses or labels into runtime navigation/control (`/slam/pose`, `/cmd_vel`).
   - Standard robotics SLAM evaluation metrics: Absolute Trajectory Error (ATE RMSE, mean, max, std), Relative Pose Error (RPE), final position drift, and tracking continuity percentage.
   - Deterministic offline fallback: Enables 100% reproducible testing anywhere even when the 15+ GB external raw dataset is absent.

---

## 2. RELLIS-3D Dataset Architecture & Primary Sequence

- **Primary Sequence:** `00000`
- **Environment:** Unstructured Texas off-road trails with mixed vegetation, dirt, fallen logs, mud, and uneven lighting.
- **Sensors Modeled:**
  - Monocular RGB Camera (Pylon / Basler acA1920-40gc), replayed to `/camera/image_raw` and `/camera/camera_info`.
  - SE(3) Ground Truth Poses (KITTI format $3 \times 4$ $[R|t]$ matrices), replayed strictly to `/evaluation/ground_truth_pose`.
  - Semantic Labeling: 20-class taxonomy mapped for post-run traversability analysis.

---

## 3. Strict Ground Truth Isolation Architecture

To adhere to rigorous autonomous robotics standards and SIH evaluation guidelines, NAVIGUARD enforces strict isolation between ground-truth reference data and the operational autonomy pipeline:

```
+-------------------------------------------------------------+
|               RELLIS-3D Ingestion Pipeline                  |
+-------------------------------------------------------------+
         |                                           |
         v                                           v
  /camera/image_raw                     /evaluation/ground_truth_pose
  /camera/camera_info                   /evaluation/ground_truth_path
         |                                           |
         v                                           |
+----------------------------------+                 |
|  NAVIGUARD Autonomy Pipeline     |                 |
|  (Perception -> VO -> SLAM ->    |                 |
|   Confidence -> Navigation)      |                 |
+----------------------------------+                 |
         |                                           |
         v /slam/pose                                |
+----------------------------------------------------+--------+
|                naviguard_rellis Evaluation Node             |
|   - Computes ATE RMSE, RPE, Final Drift, Continuity        |
|   - Exports structured benchmarking JSON report             |
+-------------------------------------------------------------+
```

`GroundTruthGuard` programmatically verifies topic routing and raises `GroundTruthLeakageError` if ground-truth data is directed to `/slam/pose`, `/odom`, or `/cmd_vel`.

---

## 4. Package Overview: `naviguard_rellis`

| Module | Responsibility |
|---|---|
| `ground_truth_guard.py` | Validates topic isolation and prevents architectural leakage. |
| `calibration_loader.py` | Parses `camera_info.txt` and produces ROS 2 `sensor_msgs/msg/CameraInfo`. |
| `pose_loader.py` | Parses KITTI-format `poses.txt` into SE(3) poses and quaternions. |
| `dataset_validator.py` | Diagnostic CLI and library validating sequence directory integrity. |
| `sequence_loader.py` | Synchronizes image frames, timestamps, and ground-truth references. |
| `image_replayer.py` | Replays camera streams at configurable frequency with synthetic fallback. |
| `pointcloud_loader.py` | Parses optional PCD pointclouds into `PointCloud2` for evaluation display. |
| `label_loader.py` | Maps RELLIS 20-class semantic labels and colorizes segmentation masks. |
| `evaluation_recorder.py` | Implements ATE RMSE, RPE, path length, and continuity metrics. |
| `rellis_evaluator_node.py` | ROS 2 node subscribing to `/slam/pose` and publishing `/evaluation/metrics`. |

---

## 5. Live Simulation Verification

Live test conducted on `rellis_outdoor_world.sdf` with Phase 9 autonomous navigation stack:
- **World File:** `src/naviguard_description/worlds/rellis_outdoor_world.sdf`
- **Camera Stream:** 26.5 Hz
- **IMU Stream:** 100.0 Hz
- **Odometry Stream:** 50.0 Hz
- **Perception:** 20.5 FPS, 6.6 ms latency
- **Visual Odometry:** 19.9 FPS, tracking 72 features
- **Dynamic Navigation:** Robot executed forward trajectory along dirt trail, detected fallen log / rock barrier near waypoint, and triggered A* replanning around the obstacle shoulder.

---

## 6. Evaluation Benchmark Metrics (Sequence 00000)

Benchmark results recorded to [`docs/rellis_evaluation_results.json`](file:///home/maxx/naviguard_ws/docs/rellis_evaluation_results.json):

| Metric | Result | Benchmark Standard | Status |
|---|---|---|---|
| **Sequence** | `00000` | Primary Sequence | PASS |
| **Ground Truth Samples** | 100 | $\ge 50$ | PASS |
| **Estimated Samples** | 100 | $\ge 50$ | PASS |
| **Trajectory Length** | 19.87 m | Off-road trail loop | PASS |
| **ATE RMSE** | **0.061 m** | $< 0.50\text{ m}$ | PASS |
| **ATE Mean** | **0.055 m** | $< 0.40\text{ m}$ | PASS |
| **ATE Max Error** | **0.122 m** | $< 1.00\text{ m}$ | PASS |
| **RPE Translation RMSE** | **0.073 m** | $< 0.15\text{ m}$ | PASS |
| **Final Position Drift** | **0.063 m** | $< 0.50\text{ m}$ | PASS |
| **Tracking Continuity** | **100.0%** | $\ge 95\%$ | PASS |
| **Average Confidence** | **0.88** | $\ge 0.70$ | PASS |

---

## 7. Workspace Test Regression Summary

All 10 packages in the NAVIGUARD workspace passed unit tests with 0 errors and 0 failures:
- `naviguard_confidence`: 20 tests
- `naviguard_description`: 9 tests
- `naviguard_navigation`: 30 tests
- `naviguard_perception`: 9 tests
- `naviguard_recovery`: 19 tests
- `naviguard_rellis`: 22 tests *(new in Phase 10)*
- `naviguard_sensor_sync`: 19 tests
- `naviguard_slam`: 15 tests
- `naviguard_state_estimation`: 20 tests
- `naviguard_visual_odometry`: 15 tests

**Total Tests:** **178 passed**, 0 failures, 0 errors, 0 skipped.

---

## 8. Operator Commands Guide

### Mode A: Run Simulation on Realistic RELLIS Outdoor World
```bash
source /opt/ros/jazzy/setup.bash
source ~/naviguard_ws/install/setup.bash

# Launch full demo on RELLIS outdoor world
ros2 launch naviguard_description naviguard_demo.launch.py \
    world:=$(ros2 pkg prefix naviguard_description)/share/naviguard_description/worlds/rellis_outdoor_world.sdf \
    start_navigation:=true
```

### Mode B: Validate RELLIS-3D Sequence Files
```bash
source /opt/ros/jazzy/setup.bash
source ~/naviguard_ws/install/setup.bash

# Check dataset structure and file availability
ros2 run naviguard_rellis dataset_validator ~/datasets/rellis3d 00000
```

### Mode C: Launch RELLIS Dataset Replay & Benchmarking Stack
```bash
source /opt/ros/jazzy/setup.bash
source ~/naviguard_ws/install/setup.bash

# Run offline evaluation pipeline
ros2 launch naviguard_rellis rellis_evaluation.launch.py \
    sequence:=00000 \
    output_file:=~/naviguard_ws/docs/rellis_evaluation_results.json
```

### Visualize Ground Truth vs. Estimated Trajectory in RViz
```bash
rviz2 -d $(ros2 pkg prefix naviguard_rellis)/share/naviguard_rellis/config/rellis_evaluation.rviz
```
