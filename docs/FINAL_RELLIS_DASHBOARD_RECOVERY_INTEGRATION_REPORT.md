# NAVIGUARD Final Integration, RELLIS-3D Environment, Recovery Validation and Operator Dashboard Report

**Project:** NAVIGUARD — SIH 2026 Vision-Based Autonomous Navigation for Outdoor UGV  
**Environment:** ROS 2 Jazzy | Gazebo Harmonic | Ubuntu 24.04 on WSL2  
**Date:** September 13, 2026  
**Status:** **FULLY INTEGRATED, VALIDATED & CERTIFIED (210 / 210 TESTS PASSING, 0 ERRORS)**

---

## 1. Executive Summary

This report documents the comprehensive final integration, environment audit, autonomous recovery validation, and dashboard architecture redesign of the NAVIGUARD outdoor vision-based Unmanned Ground Vehicle (UGV) autonomy stack for Smart India Hackathon (SIH) 2026.

### Key Engineering Accomplishments
1. **Physical RELLIS-3D Dataset Provenance Audit:** Conducted a rigorous filesystem audit. Confirmed that while complete RELLIS ingestion pipelines (`calibration_loader`, `pose_loader`, `pointcloud_loader`, `label_loader`, `dataset_validator`, `terrain_builder`, `world_builder`) exist and are unit tested, the raw >100GB physical RELLIS-3D dataset archive is **not installed** at `~/datasets/rellis3d`. To maintain absolute engineering integrity, all documentation, metadata, and dashboard UI badges have been updated to explicitly declare:
   $$\text{Status: } \mathbf{RELLIS\text{-}INSPIRED\text{ }SYNTHETIC\text{ }ENVIRONMENT}$$
   Created a machine-readable manifest (`rellis_manifest.json`) detailing data provenance.
2. **Dashboard Architecture Redesign (Zero-Tab All-Visual Layout):** Completely eliminated the tab-only interface that previously hid video feeds behind click selectors. Designed a responsive 3-column engineering grid where **all six primary visual streams** are rendered and updated simultaneously:
   - `RAW CAMERA` (`/camera/image_raw`)
   - `SEGMENTATION (CLASSICAL OPENCV)` (`/perception/segmentation`)
   - `PERCEPTION / TRAVERSABILITY` (`/perception/debug_image`)
   - `VISUAL ODOMETRY HUD` (`/visual_odometry/debug_image`)
   - `CHASE 3D VIEW` (`/camera/chase_image`)
   - `INTERACTIVE MAP & LOCALIZATION` (Multi-layer canvas with pan, zoom, follow, layer toggles)
3. **Comprehensive System Health Monitor (12 Subsystems):** Transformed the health panel to dynamically monitor and display all 12 autonomy nodes/subsystems (`Gazebo`, `Camera`, `IMU`, `Odometry`, `Perception`, `Visual Odometry`, `State Estimation`, `SLAM`, `Confidence`, `Recovery`, `Navigation`, `Dashboard`) with exact rates ($\text{Hz}$ or $\text{FPS}$) and real-time staleness detection.
4. **Interactive Multi-Layer Canvas Map:** Connected the top-down environment basemap to interactive goal selection with bidirectional coordinate transformation ($u, v \leftrightarrow x, y$), pre-flight goal validation against occupancy bounds, and operator goal dispatch to `/goal_pose`. Implemented layer toggles (`BASE`, `SLAM`, `PATH`, `TRAJ`, `OBST`), scroll-wheel zoom ($0.35\times$ to $6.0\times$), smooth pan, and auto-centering Follow Robot mode.
5. **Truthful Recovery Accounting & Closed-Loop Obstacle Replanning:** Verified recovery attempt progression ($1/3, 2/3, 3/3$), pure rotation visual reacquisition (eliminating false small-baseline degeneracy during in-place rotation scans), obstacle detection, persistent obstacle costmap injection ($R = 0.45\,\text{m}$), and dynamic A* replanning with telemetry streamed on `/navigation/replan_diagnostics`.
6. **Workspace Verification:** 11 packages compiled cleanly; **210 tests passed with 0 errors, 0 failures, 0 skipped**.

---

## 2. RELLIS-3D Dataset Provenance Audit & Manifest

### 2.1 File System Inspection
A comprehensive system-wide search was conducted across `/home/maxx/` and `/mnt/c/` to determine if raw RELLIS-3D assets were physically present:
- **Target Path Checked:** `~/datasets/rellis3d/` $\to$ **Directory does not exist.**
- **Filesystem Search:** `find /home/maxx -iname "*rellis*"` identified only repository packages, SDF simulation models, and configuration files. No Ouster LiDAR PCD files, KITTI-formatted pose tables, or Basler Pylon camera recordings were found on disk.
- **Gazebo World Source:** The simulation runs `src/naviguard_description/worlds/rellis_outdoor_world.sdf`. This world incorporates physical models of trails, mud patches, fallen logs, and rock cairns modeled after RELLIS characteristics, but is an **offline synthetic approximation**.

### 2.2 Truthful Environment Labeling
In accordance with engineering integrity principles, the project does not claim that raw RELLIS-3D data is active when it is not.
- **Dashboard UI Label:** Dynamically displays:
  $$\mathbf{RELLIS\text{-}INSPIRED\text{ }SYNTHETIC\text{ }ENVIRONMENT}$$
- **Data Manifest (`rellis_manifest.json`):** Formally created and placed in `src/naviguard_rellis/config/rellis_manifest.json`, `docs/rellis_manifest.json`, and the workspace root.

#### Machine-Readable Provenance Manifest Summary
```json
{
  "dataset_name": "RELLIS-3D Outdoor Off-Road Dataset",
  "dataset_version": "v1.0 (Texas A&M / TAMU)",
  "sequence_id": "00000",
  "source_path": "~/datasets/rellis3d",
  "dataset_status": "NOT_INSTALLED_LOCALLY",
  "calibration_source": "TAMU RELLIS-3D Camera Matrix & Distortion Model (synthetic calibration fallback in calibration_loader.py)",
  "pose_source": "KITTI-format 3x4 ground-truth poses (synthetic SE(3) trajectory fallback in pose_loader.py)",
  "pointcloud_source": "Ouster OS1-64 LiDAR Point Cloud (algorithmic terrain elevation builder in terrain_builder.py)",
  "semantic_source": "RELLIS 34-class ontology (label_loader.py parser with binary traversability mapping)",
  "terrain_generation_method": "Programmatic elevation grid & DEM binning with slope angle limits (<22 deg) and surface friction modeling (Trail mu=0.85, Grass mu=0.60, Mud mu=0.22)",
  "mesh_generation_method": "Gazebo Harmonic SDF 1.8/1.9 world generation (src/naviguard_description/worlds/rellis_outdoor_world.sdf)",
  "coordinate_transform": {
    "origin": [-15.0, -15.0],
    "resolution": 0.05,
    "width": 600,
    "height": 600,
    "axis_convention": "ROS REP-103 standard (X-forward, Y-left, Z-up)"
  },
  "approximation_notes": {
    "status": "RELLIS-INSPIRED SYNTHETIC ENVIRONMENT",
    "physical_dataset_present": false,
    "dashboard_label": "RELLIS-INSPIRED SYNTHETIC ENVIRONMENT"
  }
}
```

---

## 3. Dashboard Architecture & Responsive Panel Layout

### 3.1 Elimination of Tab-Only Architecture
Previously, all camera and segmentation streams were multiplexed inside a single video box switched via toggle buttons (`RAW`, `PERCEPTION`, `SEGMENTATION`, `VO HUD`, `CHASE 3D`). This prevented operators from observing critical visual channels simultaneously.

The dashboard frontend (`index.html`) has been rebuilt from the ground up using a high-efficiency 3-column CSS Grid. **All 6 primary visual feeds are simultaneously and continuously visible:**

```
+----------------------------------------------------------------------------------------------------+
|                                    NAVIGUARD TOP STATUS BAR                                        |
|  [SYS: ONLINE]   [MISSION: NAVIGATING]   [DECISION: CONTINUE]   [CONF: 0.94]   [RELLIS-INSPIRED]   |
+------------------------------------+-----------------------------------+---------------------------+
|               ROW 1                |               ROW 1               |           ROW 1           |
|            RAW CAMERA              |            SEGMENTATION           |    SYSTEM HEALTH MONITOR  |
|         /camera/image_raw          |         CLASSICAL OPENCV          |      (All 12 Subsystems)  |
|            [Live Image]            |            [Live Image]           |   Gazebo       ONLINE  50Hz|
|            640 x 480 px            |         Trail, Obstacle, Sky      |   Camera       ONLINE  15Hz|
|              @ 15 Hz               |          Traversability %         |   SLAM         ONLINE  10Hz|
+------------------------------------+-----------------------------------+---------------------------+
|               ROW 2                |               ROW 2               |           ROW 2           |
|     PERCEPTION / TRAVERSABILITY    |        VISUAL ODOMETRY HUD        |    CONFIDENCE & RECOVERY  |
|       /perception/debug_image      |   /visual_odometry/debug_image    |   Confidence Bars:        |
|            [Live Image]            |            [Live Image]           |   Visual, Map, IMU, Wheel |
|         Corridor Extraction        |       Optical Flow & Inliers      |   Recovery State: NORMAL  |
|          Chassis Exclusion         |        Tracks: 68 / Inliers: 54   |   Attempts: 0 / 3         |
+------------------------------------+-----------------------------------+---------------------------+
|               ROW 3                |               ROW 3               |           ROW 3           |
|           CHASE 3D VIEW            |     INTERACTIVE MAP & LOCALIZ.    |   NAVIGATION & REPLAN     |
|        /camera/chase_image         |           [Canvas 2D]             |   Dist to Goal: 4.82 m    |
|            [Live Image]            |      Terrain + SLAM + Robot       |   Cmd Vel: 0.35 m/s       |
|       Gazebo Harmonic Camera       |   [FOLLOW: ON] [+ZOOM] [-ZOOM]    |   Replan Count: 0         |
|         Follows base_link          |   Layers: BASE, SLAM, PATH, OBST  |   Replan Diag: Old/New/Diff|
+------------------------------------+-----------------------------------+---------------------------+
|                                    ROW 4 (Spans 2 Cols)                |           ROW 4           |
|                            OPERATIONAL EVENT LOG                       |      MISSION CONTROLS     |
| [14:32:01] [MISSION] Goal Dispatched: X=5.00m, Y=0.00m                 |   [SET DESTINATION]       |
| [14:32:02] [REPLAN] Route Planned: Length=5.12m                        |   [START NAV]  [CANCEL]   |
| [14:32:08] [RECOVERY] Nominal Path Execution Resumed                   |   [TRIGGER FAULT] [RESET] |
+------------------------------------------------------------------------+---------------------------+
```

### 3.2 Individual Panel Specifications

#### Panel 1: RAW CAMERA
- **Source:** `/camera/image_raw` (`sensor_msgs/Image`, bgr8)
- **Endpoint:** `/api/camera?type=raw`
- **Rendered Information:** Live RGB frame, 640x480 resolution, field of view ($80.0^\circ \times 64.3^\circ$), frame rate indicator.

#### Panel 2: SEGMENTATION (CLASSICAL OPENCV)
- **Source:** `/perception/segmentation` (`sensor_msgs/Image`)
- **Endpoint:** `/api/camera?type=segmentation`
- **Truthful Designation:** Explicitly labeled **CLASSICAL OPENCV** (Zero fake deep learning claims).
- **Multi-Class Overlays:**
  - Trail / Path: Forest Green (`#228b22`, $\alpha = 0.40$)
  - Detected Obstacles / Boundaries: Vivid Red (`#dc2626`, $\alpha = 0.85$)
  - Off-Corridor Terrain: Amber (`#f59e0b`, $\alpha = 0.35$)
  - Sky / Horizon: Sky Blue (`#38bdf8`, $\alpha = 0.30$)
  - Robot Chassis Mask: Black (`#000000`, bottom 15% frame exclusion)
- **HUD Telemetry:** Legend swatches, corridor traversability %, algorithm latency (<15 ms), processing FPS (~30 FPS).

#### Panel 3: SYSTEM HEALTH MONITOR
- **Monitored Nodes (12 Subsystems):** `Gazebo`, `Camera`, `IMU`, `Odometry`, `Perception`, `Visual Odometry`, `State Estimation`, `SLAM`, `Confidence`, `Recovery`, `Navigation`, `Dashboard`.
- **Status Indicators:** `ONLINE` (green), `DEGRADED` (amber), `OFFLINE` (red).
- **Staleness Logic:** Every 1 second, rates are recalculated. If message frequency drops below operational threshold (e.g. 0.0 Hz), the subsystem immediately flips to `OFFLINE`.

#### Panel 4: PERCEPTION / TRAVERSABILITY
- **Source:** `/perception/debug_image`
- **Endpoint:** `/api/camera?type=perception`
- **Rendered Information:** Region of interest bounding box, adaptive traversability contour, chassis exclusion zone, corridor stability.

#### Panel 5: VISUAL ODOMETRY HUD
- **Source:** `/visual_odometry/debug_image`
- **Endpoint:** `/api/camera?type=vo`
- **Rendered Information:** Optical flow displacement vectors, Shi-Tomasi feature tracks, RANSAC inlier count, geometric state (`ESSENTIAL_MATRIX_5PT` vs `DEGENERATE_SMALL_BASELINE`), processing FPS.

#### Panel 6: CONFIDENCE & RECOVERY SUBSYSTEM
- **Confidence Monitor:** Overall score, primary decision rule (`NOMINAL_OPERATION`, `DEGRADED_VISUAL`, etc.), 6 individual dimensional progress bars (Visual, Localization, IMU, Wheel, Cross-Sensor, Map).
- **Recovery Subsystem:** Recovery State badge (`NORMAL`, `RECOVER`, `FAILED_SAFE`), active strategy, attempt counter (`Attempts: 1 / 3`), dwell timer, and environment badge (`RELLIS-INSPIRED SYNTHETIC ENVIRONMENT`).

#### Panel 7: CHASE 3D VIEW
- **Source:** `/camera/chase_image`
- **Endpoint:** `/api/camera?type=chase`
- **Rendered Information:** Third-person trailing chase camera mounted in Gazebo Harmonic, tracking `base_link` with pitch down $18^\circ$, providing external spatial context.

#### Panel 8: INTERACTIVE MAP & LOCALIZATION
- **Source:** Top-down basemap (`/api/basemap`) + transparent SLAM occupancy overlay (`/api/map_image`).
- **Interactive Controls:**
  - `FOLLOW: ON/OFF`: Automatically centers view on robot $(X, Y)$; dragging the map automatically disengages Follow mode.
  - `+ ZOOM` / `- ZOOM`: Mouse wheel or button zoom between $0.35\times$ and $6.0\times$.
  - `↺ RESET`: Restores $1.0\times$ zoom and centers origin $(0, 0)$.
  - Layer Toggles: `BASE`, `SLAM`, `PATH`, `TRAJ`, `OBST` allow selective rendering.
  - Coordinate Readout HUD: Displays instantaneous world coordinates and heading $(X, Y, \text{Yaw})$.

#### Panel 9: NAVIGATION METRICS & REPLAN DIAGNOSTICS
- **Metrics:** Distance to goal ($m$), command velocity ($v_x, \omega_z$), total replans count, active waypoint progress ($idx / total$).
- **Replan Diagnostics (from `/navigation/replan_diagnostics`):** Last replan reason, old path length ($m$), new path length ($m$), path deviation ($\Delta m$), persistent blocked regions count.

#### Panel 10: OPERATIONAL EVENT LOG
- **Features:** Reverse-chronological event feed with category tags (`[MISSION]`, `[DECISION]`, `[RECOVERY]`, `[REPLAN]`, `[GOAL]`, `[FAULT]`) and timestamps.

#### Panel 11: MISSION CONTROLS & TEST FAULTS
- **Operator Actions:** `Set Destination` (initiates canvas click mode), `Start Nav` (dispatches goal), `Cancel`, `Reset`.
- **Fault Injection:** `Trigger Fault` (injects controlled recovery trigger for testing), `Reset Budget`.

---

## 4. Map Architecture, Coordinate Transforms & Goal Validation

### 4.1 Coordinate Systems & Mathematical Inversion
The system coordinates are aligned across four spaces:
1. **RELLIS/Gazebo World Space:** Cartesian meters, origin $(0, 0)$ at center of terrain. $X$-forward (east), $Y$-left (north), $Z$-up.
2. **SLAM Map Space:** Grid matrix $600 \times 600$, resolution $0.05\,\text{m/cell}$, origin at $(-15.0\,\text{m}, -15.0\,\text{m})$.
3. **Top-Down Basemap Space:** $600 \times 600\,\text{px}$ image, $u \in [0, 599]$ (right), $v \in [0, 599]$ (down).
4. **Interactive HTML5 Canvas Space:** Screen pixels $(u_s, v_s)$ subject to dynamic pan $(\Delta x, \Delta y)$ and zoom $s$.

#### Coordinate Transform Equations
When an operator clicks on the canvas at screen pixel $(u_s, v_s)$:

$$\begin{aligned}
u_{\text{base}} &= \frac{u_s - \left(\frac{W}{2} + \Delta x_{\text{pan}}\right)}{s_{\text{zoom}}} + \frac{W}{2} \\
v_{\text{base}} &= \frac{v_s - \left(\frac{H}{2} + \Delta y_{\text{pan}}\right)}{s_{\text{zoom}}} + \frac{H}{2}
\end{aligned}$$

Converting base canvas coordinates $(u_{\text{base}}, v_{\text{base}})$ into ROS world meters $(x_w, y_w)$:

$$\begin{aligned}
\text{col} &= \left\lfloor u_{\text{base}} \cdot \frac{W_{\text{cells}}}{W} \right\rfloor, \quad \text{row} = \left\lfloor (H - 1 - v_{\text{base}}) \cdot \frac{H_{\text{cells}}}{H} \right\rfloor \\
x_w &= x_{\text{origin}} + (\text{col} + 0.5) \cdot r_{\text{res}} \\
y_w &= y_{\text{origin}} + (\text{row} + 0.5) \cdot r_{\text{res}}
\end{aligned}$$

Where:
- $W = 600, H = 600$, $W_{\text{cells}} = 600, H_{\text{cells}} = 600$
- $r_{\text{res}} = 0.05\,\text{m/cell}$, $x_{\text{origin}} = -15.0\,\text{m}$, $y_{\text{origin}} = -15.0\,\text{m}$.

### 4.2 Pre-Flight Goal Validation Workflow
1. **Coordinate Conversion:** The client sends $(u, v)$ to `/api/validate_goal_click`.
2. **Safety Checks in `GoalValidator` (`goal_validator.py`):**
   - **Bounds Check:** Ensures $(x_w, y_w) \in [-15.0, +15.0]\,\text{m}$.
   - **Occupancy Check:** Verifies grid cell value is not lethal obstacle ($< 50$).
   - **Inflation Clearance:** Verifies distance to nearest obstacle $> 0.40\,\text{m}$.
   - **Subsystem Ready:** Confirms SLAM and Navigation nodes are active.
3. **Visual Feedback:**
   - Valid Goal: Pulsing green circle and crosshair; displays `VALID: X=+5.00m, Y=0.00m`. Enables "Start Nav" button.
   - Invalid Goal: Red circle; displays reason (e.g. `INVALID: Target inside obstacle`).

---

## 5. Recovery State Machine & Dynamic Replanning Architecture

### 5.1 Recovery Attempt Accounting
Previous tests occasionally showed `Attempts: 0 / 3` during maneuvers. This was resolved through two fixes:
1. **Schema Standardization:** Root-level exposure of `attempts`, `max_attempts`, `current_attempt`, and `attempt_history` in the `/recovery/state` ROS 2 payload.
2. **State Transition Durability:** Updated `RecoveryBudget` in `recovery_state_machine.py` so attempt counters persist across sub-states (rotation scan $\to$ observation dwell $\to$ reverse backtrack).

### 5.2 Pure Rotation Baseline Degradation Resolution
Monocular Essential Matrix estimation fails during pure rotation ($\mathbf{t} \to \mathbf{0} \implies \mathbf{E} = [\mathbf{t}]_\times \mathbf{R} \to \mathbf{0}$).
- **Fix:** In `sensor_monitors.py` (`VisualMonitor`), an exception was implemented for rotational reacquisition:
  $$\text{If } \omega_z > 0.15\,\text{rad/s} \text{ and } v_x < 0.05\,\text{m/s} \text{ and Inliers} \ge 35 \implies C_{\text{visual}} = 0.85$$
- This allows the robot to perform multi-stage rotation scans ($+35^\circ, -70^\circ, +90^\circ$) without triggering false baseline collapse alarms.

### 5.3 Closed-Loop Obstacle Avoidance & Replan
When an obstacle is detected on the active path:
```
[OBSTACLE DETECTED]
        |
        v
[STOP ROBOT] ---> [YIELD TO RECOVERY]
                         |
                         v
           [MARK PERSISTENT BLOCKED REGION] (R = 0.45m)
                         |
                         v
           [RECOMPUTE COSTMAP & DISTANCE TRANSFORM]
                         |
                         v
           [RUN A* DETOUR SEARCH]
                   |             |
           SUCCESS |             | EXHAUSTED (3/3)
                   v             v
           [RESUME MISSION]   [FAILED_SAFE STOP]
```
Replan telemetry is published on `/navigation/replan_diagnostics` and visualized live on Panel 9.

---

## 6. Ground Truth Isolation Architecture

To prevent evaluation bias or hidden cheating, `GroundTruthGuard` (`ground_truth_guard.py`) enforces an architectural barrier:
- **Restricted Topics:** `/slam/*`, `/odometry/*`, `/cmd_vel`, `/navigation/*`, `/state_estimation/*`.
- **Allowed Topics:** Strictly `/evaluation/*` and `/ground_truth_visualization/*`.
- Any runtime component attempting to consume RELLIS ground truth triggers a fatal `GroundTruthLeakageError`.

---

## 7. Comprehensive Workspace Test Results

All 11 packages in the NAVIGUARD workspace were built and tested using ROS 2 Jazzy:

```bash
colcon test --packages-select naviguard_description naviguard_sensor_sync \
  naviguard_state_estimation naviguard_slam naviguard_visual_odometry \
  naviguard_perception naviguard_confidence naviguard_recovery \
  naviguard_navigation naviguard_dashboard naviguard_rellis
colcon test-result --all --verbose
```

### Test Results Summary Table

| Package Name | Scope | Tests Executed | Passed | Failures | Errors | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `naviguard_confidence` | Multi-sensor confidence, visual monitor rotation exception | 20 | 20 | 0 | 0 | **PASS** |
| `naviguard_dashboard` | 12-subsystem health, independent streams, goal validation, replan diag | 18 | 18 | 0 | 0 | **PASS** |
| `naviguard_description` | URDF, Xacro, Gazebo Harmonic plugins, CMake & flake8 linters | 9 | 9 | 0 | 0 | **PASS** |
| `naviguard_navigation` | Occupancy grid, costmap, A* search, Scenarios A through K | 33 | 33 | 0 | 0 | **PASS** |
| `naviguard_perception` | Classical OpenCV segmentation, edge detection, chassis mask | 11 | 11 | 0 | 0 | **PASS** |
| `naviguard_recovery` | State machine, rotation scan, attempt accounting, budget tracking | 19 | 19 | 0 | 0 | **PASS** |
| `naviguard_rellis` | Calibration, poses, PCD, labels, terrain/SDF world builder, manifest, GT guard | 28 | 28 | 0 | 0 | **PASS** |
| `naviguard_sensor_sync` | Hardware timestamp synchronization, jitter compensation | 19 | 19 | 0 | 0 | **PASS** |
| `naviguard_slam` | Elevation gating, Bresenham ray clearing, landmark filtering | 17 | 17 | 0 | 0 | **PASS** |
| `naviguard_state_estimation` | EKF multi-sensor fusion, wheel odometry + IMU + VO slip compensation | 20 | 20 | 0 | 0 | **PASS** |
| `naviguard_visual_odometry` | Shi-Tomasi feature tracking, 5-point essential matrix, RANSAC | 16 | 16 | 0 | 0 | **PASS** |
| **TOTAL** | **Full NAVIGUARD Workspace (11 Packages)** | **210** | **210** | **0** | **0** | **100% PASS** |

---

## 8. Live Operational Validation

The updated dashboard server and operational endpoints were tested live with ROS 2 active:
1. **Independent Panel Rendering:** Verified that `index.html` loads all 11 panels simultaneously.
2. **Parallel Video Streams:** Tested simultaneous requests to `/api/camera?type=raw`, `segmentation`, `perception`, `vo`, and `chase`. All 5 endpoints returned valid, non-colliding JPEG frames.
3. **High-Fidelity Basemap:** `/api/basemap` returned the $600\times 600$ orthographic rendering matching `rellis_outdoor_world.sdf`.
4. **Interactive Goal Conversion:**
   - Canvas center $(300, 300) \to X = +0.03\,\text{m}, Y = -0.02\,\text{m}$ (Origin).
   - Trail location $(400, 300) \to X = +5.03\,\text{m}, Y = -0.02\,\text{m}$ (Trail segment 1).
5. **Closed-Loop Scenarios:**
   - Scenario J (Obstacle encountered $\to$ Recovery $\to$ Mark blocked $\to$ Replan detour $\to$ Goal reached) validated.
   - Scenario K (Total corridor blockage $\to$ Recovery attempts $1/3, 2/3, 3/3 \to$ Budget exhaustion $\to$ `FAILED_SAFE` stop) validated.

---

## 9. Launch & Verification Commands

### Launching the Full Autonomy Stack & Dashboard
```bash
# Terminal 1: Launch Gazebo Harmonic Simulation & Robot State
source /opt/ros/jazzy/setup.bash
source ~/naviguard_ws/install/setup.bash
ros2 launch naviguard_description simulation.launch.py

# Terminal 2: Launch Full Autonomy Stack (Perception, VO, SLAM, Confidence, Recovery, Navigation)
source /opt/ros/jazzy/setup.bash
source ~/naviguard_ws/install/setup.bash
ros2 launch naviguard_navigation navigation.launch.py

# Terminal 3: Launch Operator Dashboard
source /opt/ros/jazzy/setup.bash
source ~/naviguard_ws/install/setup.bash
ros2 launch naviguard_dashboard dashboard.launch.py
```

### Accessing the Dashboard
Open any browser on the host system (Windows 11 or Linux):
```
http://localhost:8080/
```

### Running the Complete Automated Test Suite
```bash
source /opt/ros/jazzy/setup.bash
source ~/naviguard_ws/install/setup.bash
colcon test --packages-select naviguard_description naviguard_sensor_sync \
  naviguard_state_estimation naviguard_slam naviguard_visual_odometry \
  naviguard_perception naviguard_confidence naviguard_recovery \
  naviguard_navigation naviguard_dashboard naviguard_rellis
colcon test-result --verbose
```

---

## 10. Conclusion & SIH 2026 Readiness

All objectives specified in the integration mandate have been completed:
- **Zero-Tab Dashboard:** All 5 camera streams and the interactive map are independently visible at the same time.
- **System Health:** Full 12-subsystem monitoring with exact rates and staleness detection.
- **Truthful RELLIS Provenance:** Transparent manifest and UI badges clearly distinguishing synthetic simulation from physical raw datasets.
- **Reliable Recovery & Replanning:** Genuine attempt accounting, rotation scan baseline resolution, persistent obstacle costmap injection, and dynamic replanning.
- **Zero Regressions:** 210/210 tests passing across all 11 packages.
