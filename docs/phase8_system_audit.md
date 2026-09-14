# Phase 8 System Audit: NAVIGUARD Robotics Foundation

**Project**: NAVIGUARD — Confidence-Aware Closed-Loop Navigation & Autonomous Recovery for an Outdoor UGV  
**Auditor**: NAVIGUARD Lead Robotics Engineer  
**Date**: September 11, 2026  
**Environment**: Ubuntu 24.04 Noble, ROS 2 Jazzy, Gazebo Harmonic, ros_gz_bridge, WSL2  

---

## 1. Existing Architecture & Verified Packages

The workspace `/home/maxx/naviguard_ws` currently contains 7 active packages representing Phases 1 through 7:

1. `naviguard_description`: 4-wheel differential-drive UGV URDF/Xacro, sensor placements (RGB camera, 6-DoF IMU, wheel encoders), physics models, Gazebo Harmonic world (`naviguard_world.sdf`), and `ros_gz_bridge` integration.
2. `naviguard_perception`: Deterministic outdoor image processing, adaptive thresholding, ground-region segmentation, obstacle edge filtering, and `/perception/debug_image`.
3. `naviguard_visual_odometry`: Shi-Tomasi feature tracking, pyramidal Lucas-Kanade optical flow, forward-backward validation, median absolute deviation (MAD) filtering, epipolar 5-point essential matrix estimation with RANSAC, cheirality gating, `/visual_odometry/telemetry`, and `/visual_odometry/debug_image`.
4. `naviguard_sensor_sync`: Inter-sensor timestamp monitoring, circular sliding buffers, jitter estimation, cross-sensor delay diagnostics, and `/sensor_sync/diagnostics`.
5. `naviguard_state_estimation`: Extended Kalman Filter (EKF) fusing wheel odometry, IMU angular rate & linear acceleration, and scale-gated monocular visual displacement. Publishes `/state_estimation/odom` and `/state_estimation/diagnostics`.
6. `naviguard_slam`: Visual-Inertial Keyframe Graph-SLAM, ORB feature extraction, 2D occupancy grid mapping, place recognition, Levenberg-Marquardt pose graph optimization (PGO), loop closure detection, and map serialization services (`/slam/save_map`, `/slam/load_map`).
7. `naviguard_confidence`: Multi-dimensional confidence engine monitoring 7 dimensions (visual, localization, IMU, wheel, temporal, cross-sensor, map) and finite state machine with debounced hysteresis publishing `/naviguard/decision`, `/naviguard/diagnostics`, and `/naviguard/decision_marker`.

---

## 2. Relevant Topics

### Motion Control
- `/cmd_vel` (`geometry_msgs/msg/Twist`): Bridges ROS 2 to Gazebo Harmonic diff-drive plugin (`direction: ROS_TO_GZ`). Currently unmanaged by any autonomous node.

### Odometry & Localization
- `/odom` (`nav_msgs/msg/Odometry`): Differential-drive wheel odometry from Gazebo bridge (~50 Hz).
- `/state_estimation/odom` (`nav_msgs/msg/Odometry`): EKF fused state estimate (~20 Hz).
- `/slam/pose` (`geometry_msgs/msg/PoseWithCovarianceStamped`): 2D map-referenced pose estimate (~10 Hz).
- `/slam/trajectory` (`nav_msgs/msg/Path`): Estimated keyframe path history.
- `/slam/map` (`nav_msgs/msg/OccupancyGrid`): 2D local occupancy grid map ($30\text{m}\times 30\text{m}$, 5 cm resolution).

### Sensors & Bridges
- `/camera/image_raw` (`sensor_msgs/msg/Image`, Best-Effort QoS, 640x480 mono8/rgb8, ~10–12 Hz).
- `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`, Reliable QoS, intrinsics $f_x, f_y, c_x, c_y$).
- `/imu` (`sensor_msgs/msg/Imu`, Best-Effort QoS, ~60 Hz).
- `/joint_states` (`sensor_msgs/msg/JointState`).
- `/clock` (`rosgraph_msgs/msg/Clock`).

### Diagnostics & Decision Telemetry
- `/visual_odometry/telemetry` (`diagnostic_msgs/msg/DiagnosticArray`): Tracks, inliers, flow magnitude, essential matrix validity.
- `/sensor_sync/diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`): Overall sync status (`PASS`/`WARN`/`FAIL`), latencies, jitter.
- `/state_estimation/diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`): Consistency flags, yaw rate residuals, Mahalanobis rejections.
- `/slam/diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`): Tracking state (`OK`/`DEGRADED`/`TRACKING_LOST`), reprojection error, landmark count.
- `/naviguard/decision` (`std_msgs/msg/String`): Real-time JSON payload containing:
  - `state`: `"CONTINUE"`, `"VERIFY"`, `"RECOVER"`
  - `state_code`: `0`, `1`, `2`
  - `overall_confidence`: float $\in [0.0, 1.0]$
  - `primary_reason`: string explanation
  - `secondary_reasons`: list of string explanations
  - `dwell_time_sec`: duration in current state
  - `scores`: dictionary of 7 individual dimension scores
- `/naviguard/diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`): Standard ROS 2 diagnostics tree.
- `/naviguard/decision_marker` (`visualization_msgs/msg/MarkerArray`): Color-coded 3D halo and text banner.

---

## 3. Relevant Services
- `/slam/save_map` (`std_srvs/srv/Trigger`): Saves occupancy grid and keyframe graph to disk (JSON/YAML/PGM).
- `/slam/load_map` (`std_srvs/srv/Trigger`): Loads saved map for localization mode.

---

## 4. Relevant TF Frames

Authoritative coordinate frame tree:
```
map
 └── odom (published by naviguard_slam)
      └── base_footprint (published by diff-drive plugin or state estimator)
           └── base_link
                ├── chassis_link
                ├── camera_link
                ├── imu_link
                ├── left_wheel_front_link
                ├── left_wheel_rear_link
                ├── right_wheel_front_link
                └── right_wheel_rear_link
```

---

## 5. Existing Motion-Control Path & Navigation Capability

- **Current Motion Command Path**: External publisher $\to$ `/cmd_vel` $\to$ `ros_gz_bridge` $\to$ Gazebo diff-drive plugin.
- **Planner / Waypoint Navigation**: **NONE**. No path planning, waypoint following, or global planner node exists in the workspace.
- **Nav2 Status**: **NOT INSTALLED**. Nav2 packages do not exist in `/opt/ros/jazzy/share` or in the workspace.
- **Emergency Stop / Safe Stop**: **NONE**. Prior to Phase 8, when a failure was detected by Phase 7 (`RECOVER`), no node commanded zero velocity or intervened in robot motion.

---

## 6. Architectural Gaps Phase 8 Must Implement

To achieve closed-loop autonomous recovery without introducing an entire out-of-scope navigation stack or fake demos:

1. **Autonomous Recovery Node (`naviguard_recovery`)**:
   - Subscribes to `/naviguard/decision`, `/naviguard/diagnostics`, `/slam/pose`, `/slam/map`, `/state_estimation/odom`.
   - Owns the authoritative velocity command output to `/cmd_vel` during recovery interventions.
2. **Deterministic Recovery Finite-State Machine**:
   - Explicit states: `NORMAL`, `VERIFY`, `SAFE_STOP`, `SELECT_CHECKPOINT`, `RECOVER`, `RELOCALIZE`, `VERIFY_RECOVERY`, `REPLAN`, `RESUME`, `FAILED_SAFE`.
3. **Active Safe Stop Mechanism**:
   - Immediate command of $v_x=0.0, \omega_z=0.0$ on entering `SAFE_STOP`, sustained for a safety interval, preventing upstream override.
4. **Trusted State / Checkpoint Manager**:
   - Continuously buffers high-confidence poses ($C_{\text{overall}} \ge 0.85$) while in `CONTINUE`.
   - Multi-criteria scoring (confidence, map clearance, proximity, recency) to pick the optimal backtrack target.
   - Strictly forbidden from using ground truth / simulator poses.
5. **Closed-Loop Recovery Actions**:
   - `STOP_AND_RELOCALIZE`: Stationary dwell while checking sensor convergence.
   - `ROTATE_FOR_VISUAL_REACQUISITION`: Slow, bounded angular rotation ($\le 45^\circ$, bounded rate $\le 0.3\text{ rad/s}$) to bring textured landmarks into camera FOV.
   - `SHORT_BACKTRACK`: Bounded, map-cleared reverse motion toward the selected trusted checkpoint.
6. **Relocalization & State Verification**:
   - Multi-frame verification dwell (requiring sustained confidence recovery before declaring `RECOVERY_SUCCESS`).
7. **Replan & Resume Interface**:
   - Clean abstraction connecting recovery completion to path resuming (ready for future Phase 9 navigation).
8. **Bounded Recovery Budget**:
   - Hard limits on recovery attempts, duration, backtrack distance, and rotation; transition to `FAILED_SAFE` upon budget exhaustion.
