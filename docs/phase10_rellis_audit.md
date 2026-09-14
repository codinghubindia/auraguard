# NAVIGUARD Phase 10 — RELLIS-3D Dataset & Integration Audit

**Date**: September 2026  
**Engineer**: NAVIGUARD RELLIS-3D Integration & Evaluation Lead  
**Workspace**: `~/naviguard_ws`  
**Dataset Reference**: RELLIS-3D: Data, Assessment, and Empirical Evaluation for Outdoor Unstructured Environment (IEEE RA-L / IROS 2021)  

---

## 1. Official Dataset Specifications

| Attribute | Specification |
| :--- | :--- |
| **Data Collection Platform** | Clearpath Warthog UGV (4-wheel off-road skid-steer platform) |
| **Primary Location** | Texas A&M University RELLIS Campus (unpaved roads, trails, grass, woods, puddles) |
| **Camera Sensor** | Basler acA1920-50gc (RGB, 1920x1200 resolution, Bayer RG8 / RGB8, ~10 Hz) |
| **Stereo Camera** | FLIR Blackfly S (Stereo baseline ~0.30 m) |
| **LiDAR Sensor** | Ouster OS1-64 (64 channels, 1024/2048 horizontal resolution, 10 Hz) |
| **GNSS/INS Reference** | VectorNav VN-300 Dual-Antenna GNSS/INS (RTK-corrected poses, 100 Hz IMU) |
| **Semantic Labels** | 20 outdoor terrain classes (per-pixel image masks and 3D point cloud annotations) |
| **Licensing** | Creative Commons Attribution 4.0 International (CC BY 4.0) |

---

## 2. Sequence Inventory & Selection

| Sequence ID | Length (Frames) | Approximate Path Length | Dominant Terrain & Features | Suitability Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **`00000`** | **6,235** | **~1,350 m** | **Gravel path, dirt trail, grass margins, tree canopy, natural curves** | **SELECTED as `RELLIS_PRIMARY_SEQUENCE`**: Benchmark reference with highest visual continuity, stable illumination, and clean off-road trail boundaries. |
| `00001` | 3,343 | ~750 m | Deep puddles, slick mud, tall grass, steep turns | Stress case for severe traction slip and visual degradation. |
| `00002` | 4,012 | ~850 m | Wide open field, uniform dry grass, minimal landmarks | High risk of visual odometry drift due to low feature contrast. |
| `00003` | 5,120 | ~1,100 m | Deep wooded trail, dense shadows, high dynamic range | Significant lighting variation. |
| `00004` | 2,980 | ~620 m | Obstacle-dense off-road track, logs, barriers, bushes | Excellent for collision avoidance and narrow passage evaluation. |

### Primary Sequence Selection Rationale
Sequence **`00000`** is selected as `RELLIS_PRIMARY_SEQUENCE`:
1. **Visual Continuity**: Continuous off-road trail with well-defined trail boundaries, enabling reliable evaluation of the monocular perception edge and ground segmentation pipelines.
2. **Feature Density**: Trees, logs, and ground textures provide rich Shi-Tomasi / ORB visual features for visual odometry and SLAM benchmarking.
3. **Traversability Dynamics**: Combines firm dirt/gravel with softer grass shoulders, representative of outdoor UGV patrol missions.

---

## 3. Required File Layout per Sequence

An official RELLIS-3D sequence directory consists of:
```
<RELLIS_DATA_ROOT>/00000/
├── pcd/                       # 64-beam LiDAR point clouds (.pcd format)
├── pcd_annotated/             # Semantically annotated point clouds (.pcd format)
├── pcd_timestamp.txt          # Microsecond timestamps for each LiDAR frame
├── camera_info.txt            # Camera intrinsics (K matrix, distortion, image dimensions)
├── transforms.yaml            # Sensor extrinsic transformations (LiDAR <-> Camera <-> Body)
├── poses.txt                  # 4x4 SE(3) poses in KITTI format (12 floats per line)
└── gps_ins.txt                # Raw and RTK-filtered GNSS/INS state estimates
```

---

## 4. Expected Disk Usage & Storage Management

* **Full Dataset (5 Sequences, raw LiDAR + images)**: ~85–110 GB.
* **Single Sequence Raw (`00000`)**: ~15–18 GB.
* **Camera Images + Poses + Calibration Only**: ~1.5–2.5 GB.
* **Synthetic Evaluation & Unit Test Fixture**: ~2.5 MB (bundled in package tests for zero-dependency CI).
* **Storage Location Policy**:
  * External configurable root: `~/datasets/rellis3d/` (configurable via `RELLIS_DATA_ROOT` environment variable or ROS parameter).
  * No large raw dataset files committed inside `~/naviguard_ws/src/`.
  * Generated collision/visual models stored in: `~/naviguard_ws/data/rellis_generated/`.

---

## 5. Ground-Truth Boundaries: Allowed vs. Forbidden Consumption

To prevent data leakage and guarantee that NAVIGUARD remains a genuine autonomous system:

| RELLIS Data Field | Allowed in Autonomous Navigation? | Permitted System Use | Violation Condition |
| :--- | :--- | :--- | :--- |
| **Camera Images** (`pcd/` or RGB images) | **YES** (Sensor Input) | Published to `/camera/image_raw` via replay adapter | None (legitimate sensor input) |
| **Camera Calibration** (`camera_info.txt`) | **YES** (Intrinsics) | Published to `/camera/camera_info` | None (standard sensor calibration) |
| **Poses** (`poses.txt`) | **STRICTLY FORBIDDEN** | Offline evaluation only (ATE RMSE, RPE, RViz GT trajectory) | Leakage into `/slam/pose`, `/odom`, `/cmd_vel` |
| **GNSS/INS** (`gps_ins.txt`) | **STRICTLY FORBIDDEN** | Offline evaluation only | Using GPS coordinates to navigate or correct SLAM |
| **Semantic Labels** | **FORBIDDEN for direct control** | Offline perception evaluation and visual display | Using ground-truth class masks to steer robot |
| **Pre-recorded Path** | **STRICTLY FORBIDDEN** | Evaluation comparison | Hard-coding recorded path into A* planner |
