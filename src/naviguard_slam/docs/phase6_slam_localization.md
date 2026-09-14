# NAVIGUARD Phase 6: Visual-Inertial SLAM & Metric Localization

## 1. Selected SLAM Architecture & Selection Reasoning

### Selection Rationale
In accordance with the project requirements:
- **Environment constraints**: Ubuntu 24.04 LTS Noble, ROS 2 Jazzy, Gazebo Harmonic, monocular RGB camera, 6-axis IMU, differential-drive wheel odometry.
- **Dependency analysis**: Popular external SLAM frameworks (such as ORB-SLAM3, VINS-Mono, RTAB-Map) either lack native ROS 2 Jazzy binaries on Ubuntu 24.04 Noble without root access or require massive, fragile source-dependency trees (Ceres 2.2, DBoW2, Pangolin, g2o) that frequently fail to compile on GCC 13.
- **Architectural Solution**: A modular, mathematically sound **Visual-Inertial Keyframe Graph-SLAM** system was designed in `naviguard_slam` using maintained, native ROS 2 / SciPy / OpenCV libraries.
  - **Visual Front-End**: OpenCV ORB feature extraction and multi-view triangulation.
  - **Metric Prior & Scale Integration**: Phase 5B metric state estimator constraints ($\Delta T_{i, i+1}$) solving monocular scale ambiguity.
  - **Place Recognition Back-End**: Fast Hamming-distance matching with spatial search gating and RANSAC geometric verification.
  - **Pose Graph Optimization (PGO)**: Non-linear least-squares optimization over SE(2) pose graph edges with robust Huber loss (`scipy.optimize.least_squares`).
  - **Dual Map Generation**: Sparse 3D visual landmark database + 2D navigation occupancy grid (`nav_msgs/msg/OccupancyGrid`).

---

## 2. Package Architecture

```
src/naviguard_slam/
├── package.xml
├── setup.py
├── setup.cfg
├── config/
│   ├── slam_params.yaml
│   └── naviguard_slam.rviz
├── launch/
│   └── slam.launch.py
├── naviguard_slam/
│   ├── __init__.py
│   ├── keyframe.py
│   ├── landmark_database.py
│   ├── occupancy_grid_mapper.py
│   ├── place_recognition.py
│   ├── pose_graph_optimizer.py
│   ├── slam_diagnostics.py
│   └── slam_node.py
├── worlds/
│   └── naviguard_slam_world.sdf
├── maps/
│   ├── naviguard_slam_map.json
│   ├── naviguard_slam_map.yaml
│   └── naviguard_slam_map.pgm
├── test/
│   ├── test_keyframe_and_landmarks.py
│   ├── test_occupancy_grid.py
│   ├── test_pose_graph_optimizer.py
│   ├── test_place_recognition.py
│   ├── test_tracking_failure_and_diagnostics.py
│   ├── test_map_save_load.py
│   └── test_slam_node_lifecycle.py
└── docs/
    ├── phase6_slam_localization.md
    └── slam_verification_results.json
```

---

## 3. Sensor Interfaces & Frame Architecture

### Subscribed Topics
- `/camera/image_raw` (`sensor_msgs/msg/Image`, Best Effort / sensor_data QoS)
- `/camera/camera_info` (`sensor_msgs/msg/CameraInfo`, Reliable QoS)
- `/state_estimation/odom` (`nav_msgs/msg/Odometry`, Reliable QoS)
- `/imu` (`sensor_msgs/msg/Imu`, Best Effort / sensor_data QoS)

### Published Topics
- `/slam/pose` (`geometry_msgs/msg/PoseWithCovarianceStamped`): Global robot pose in `map` frame.
- `/slam/trajectory` (`nav_msgs/msg/Path`): Keyframe trajectory path in `map` frame.
- `/slam/map` (`nav_msgs/msg/OccupancyGrid`): 2D occupancy grid (0.05m resolution).
- `/slam/landmarks` (`visualization_msgs/msg/MarkerArray`): 3D visual landmark point cloud.
- `/slam/diagnostics` (`diagnostic_msgs/msg/DiagnosticArray`): Comprehensive SLAM telemetry and health status.

### TF Hierarchy
The system strictly enforces REP-105 standard:
$$\text{map} \longrightarrow \text{odom} \longrightarrow \text{base\_link} \longrightarrow \text{camera\_link} \longrightarrow \text{camera\_optical\_link}$$
$$\text{base\_link} \longrightarrow \text{imu\_link}$$

- **`odom -> base_link`**: Broadcast by Gazebo differential-drive plugin.
- **`map -> odom`**: Broadcast authoritatively by `slam_node`. Computed as:
$$\theta_{m \to o} = \text{wrap}(\theta_{map} - \theta_{odom})$$
$$x_{m \to o} = x_{map} - (x_{odom} \cos\theta_{m \to o} - y_{odom} \sin\theta_{m \to o})$$
$$y_{m \to o} = y_{map} - (x_{odom} \sin\theta_{m \to o} + y_{odom} \cos\theta_{m \to o})$$
This ensures no conflicting or competing transform broadcasts occur on `/tf`.

---

## 4. Operating Modes

### Mode A: MAPPING
1. Starts at origin $(0, 0, 0)$ in `map` frame.
2. Creates keyframes on motion threshold: $\Delta d \ge 0.35\text{ m}$ or $\Delta\theta \ge 0.25\text{ rad}$.
3. Triangulates 3D visual landmarks from consecutive keyframe stereo baselines.
4. Updates 2D occupancy grid free space along trajectory and marks obstacles.
5. Continuously evaluates loop-closure revisits against past keyframes.
6. When loop closure is found, solves pose graph optimization and distributes correction along trajectory.

### Mode B: LOCALIZATION
1. Loads serialized map package from disk (`.json` landmark graph + `.yaml`/`.pgm` grid).
2. Matches camera visual features against global landmark database using PnP / RANSAC.
3. Continuously corrects `map -> odom` transform to track robot pose within the saved map.
4. If feature count drops below threshold, flags `DEGRADED` or `TRACKING_LOST` without crashing.

---

## 5. Map Persistence Formats

The map is saved as a complete package:
1. **JSON Landmark & Graph Database** (`naviguard_slam_map.json`): Keyframe poses, timestamps, 3D landmark coordinates, and descriptors.
2. **Standard ROS Occupancy Grid Image** (`naviguard_slam_map.pgm`): Binary P5 grayscale image representing occupancy costs.
3. **Map Metadata YAML** (`naviguard_slam_map.yaml`): Standard ROS map_server format with resolution, origin, and thresholds.

Saved via `/slam/save_map` ROS 2 service call:
```bash
ros2 service call /slam/save_map std_srvs/srv/Trigger
```

---

## 6. Scope Boundaries & Important Notice

> [!IMPORTANT]
> Phase 6 establishes visual SLAM and metric localization. It does NOT implement autonomous navigation, Nav2 path planning, or NAVIGUARD autonomous recovery behaviors.
