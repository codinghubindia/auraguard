# NAVIGUARD Phase 9 — Pre-Implementation System Audit

**Date**: September 2026  
**Auditor**: NAVIGUARD Mission & Global Navigation Engineer  
**Workspace**: `~/naviguard_ws`  
**Environment**: Ubuntu 24.04 Noble, ROS 2 Jazzy, Gazebo Harmonic, WSL2  

---

## 1. Audit Questions & Verified Findings

### 1. Which frame is the global navigation frame?
* **Verified Frame**: `map`
* **Authority**: `slam_node` (in `naviguard_slam`) publishes the authoritative coordinate frame `map` and broadcasts the dynamic transform `map -> odom`.

### 2. Which frame represents the robot?
* **Verified Frames**: `base_link` and `base_footprint`.
* **Details**: `base_footprint` is the ground-plane projection $(z=0)$; `base_link` is the robot chassis coordinate frame. Robot pose published on `/slam/pose` is in the `map` frame with child orientation corresponding to `base_link`.

### 3. What occupancy-grid message is published?
* **Topic**: `/slam/map`
* **Message Type**: `nav_msgs/msg/OccupancyGrid`
* **QoS**: Transient local or reliable, published by `slam_node` (`OccupancyGridMapper`).

### 4. What map resolution is used?
* **Resolution**: `0.05` meters/cell ($5\text{ cm}$ per grid cell).
* **Source**: Defined in `slam_params.yaml` (`map_resolution: 0.05`).

### 5. What map origin is used?
* **Origin**: $x_0 = -15.0\text{ m}, y_0 = -15.0\text{ m}, z_0 = 0.0\text{ m}$.
* **Grid Dimensions**: $600 \times 600$ cells covering a $30.0\text{ m} \times 30.0\text{ m}$ spatial area.
* **Cell Semantics**:
  * `-1`: Unknown space
  * `0`: Free space (cleared along robot path within $0.35\text{ m}$ radius)
  * `100`: Occupied space (projected 3D visual obstacle landmarks)

### 6. How is the robot pose obtained?
* **Topic**: `/slam/pose` (`geometry_msgs/msg/PoseStamped`) in the `map` frame.
* **TF Tree**: `map -> odom -> base_footprint -> base_link`.
* **State Estimation**: `/state_estimation/odom` (`nav_msgs/msg/Odometry`) in `odom` frame at $20\text{ Hz}$ for high-frequency velocity and differential motion feedback.

### 7. Who currently publishes `/cmd_vel`?
* **Bridge**: `ros_gz_bridge` translates ROS `/cmd_vel` (`geometry_msgs/msg/Twist`) to Gazebo Harmonic `gz.msgs.Twist`.
* **Publishers**:
  * `naviguard_recovery`: `recovery_node` publishes velocity commands during recovery maneuvers (`SAFE_STOP`, `RECOVER`).
  * `naviguard_recovery`: `naviguard_demo_motion` publishes demo trajectory commands during standalone demo mode (disabled when navigation is running).

### 8. How does recovery take ownership of `/cmd_vel`?
* `recovery_node` publishes its lifecycle state on `/recovery/state` (`std_msgs/msg/String`, JSON format).
* When state is `SAFE_STOP`, `RECOVER`, or `FAILED_SAFE`, `recovery_node` publishes commands on `/cmd_vel`.
* When state is `NORMAL` or `VERIFY`, `recovery_node` does not publish to `/cmd_vel`.
* `naviguard_navigation` will subscribe to `/recovery/state` and `/naviguard/decision`. When recovery is triggered, navigation immediately yields `/cmd_vel` and transitions to `RECOVERY_WAIT`.

### 9. Does a planner already exist?
* **Verified**: No planner exists anywhere in the workspace. There is currently no global path planner, no waypoint follower, and no goal interface.

### 10. Is Nav2 installed?
* **Verified**: Nav2 is **NOT** installed in `/opt/ros/jazzy`.
* The mission & global navigation package must be a self-contained ROS 2 implementation.

---

## 2. Robot Footprint & Obstacle Inflation Analysis
* **Chassis Length**: $0.50\text{ m}$
* **Chassis Width**: $0.32\text{ m}$
* **Track Width**: $0.42\text{ m}$ (center-to-center) + wheel width $0.06\text{ m}$ = $0.48\text{ m}$ total width.
* **Circumscribed Radius**: $R_{circ} = \sqrt{(0.25)^2 + (0.24)^2} \approx 0.35\text{ m}$.
* **Inscribed Radius**: $R_{inscr} = 0.24\text{ m}$.
* **Recommended Inflation Radius**: $R_{inflate} = R_{circ} + \text{margin} = 0.35\text{ m} + 0.15\text{ m} = 0.50\text{ m}$ ($10$ grid cells at $0.05\text{ m}$ resolution).
