# NAVIGUARD Final Integration, Operator Dashboard & Demonstration Architecture

**Project:** NAVIGUARD — Vision-Based Autonomous Navigation for UGV in Outdoor Unstructured Environments  
**Competition:** Smart India Hackathon (SIH) 2026 — Problem Statement 26126  
**Final Release:** Master Operator Dashboard & One-Command System Integration

---

## 1. System Architecture

The complete NAVIGUARD pipeline operates in a closed loop, maintaining strict camera-primary autonomy:

```
+------------------------------------------------------------------------------------+
|                         GAZEBO HARMONIC (RELLIS OUTDOOR WORLD)                     |
|  - Off-Road Dirt Trail | Vegetation & Trees | Mud Terrain | Natural Log Barriers   |
+------------------------------------------------------------------------------------+
       | /camera/image_raw (26 Hz)              | /imu (100 Hz)        | /odom (50 Hz)
       v                                         v                      v
+------------------------+             +---------------------------------------------+
| naviguard_perception   |             |           naviguard_sensor_sync             |
| (Traversability/Edges) |             |  (Cross-Sensor Timestamp Alignment)         |
+------------------------+             +---------------------------------------------+
       |                                         |
       v                                         v
+-------------------------------+      +---------------------------------------------+
|  naviguard_visual_odometry    | ---> |         naviguard_state_estimation          |
|  (LK Optical Flow + RANSAC)   |      |        (Multi-Rate Metric EKF Fusion)       |
+-------------------------------+      +---------------------------------------------+
       |                                         |
       v                                         v
+------------------------------------------------------------------------------------+
|                                 naviguard_slam                                     |
|           (Visual-Inertial Localization + Metric 2D Occupancy Grid)                |
+------------------------------------------------------------------------------------+
       | /slam/pose, /slam/map, /slam/trajectory
       v
+------------------------------------------------------------------------------------+
|                               naviguard_confidence                                 |
|             (Multi-Dimensional Evaluation: Visual, IMU, Loc, Map)                  |
|                 Produces: CONTINUE  |  VERIFY  |  RECOVER                          |
+------------------------------------------------------------------------------------+
       | /naviguard/decision
       v
+------------------------------------------------------------------------------------+
|                               naviguard_recovery                                   |
|   (FSM: SAFE_STOP -> ROTATE_REACQUISITION / BACKTRACK -> RELOCALIZE -> RESUME)     |
+------------------------------------------------------------------------------------+
       | Arbitration
       v
+------------------------------------------------------------------------------------+
|                               naviguard_navigation                                 |
|          (Mission Management, A* Global Planner, Dynamic Replanning)               |
+------------------------------------------------------------------------------------+
       | /cmd_vel (ONLY navigation and recovery publish motor commands)
       v
+------------------------------------------------------------------------------------+
|                                    UGV MOTORS                                      |
+------------------------------------------------------------------------------------+
                                       ^
                                       | /goal_pose (PoseStamped)
+------------------------------------------------------------------------------------+
|                              naviguard_dashboard                                   |
|   (Web Server on Port 8080 | Telemetry Bridge | Interactive Click-to-Goal Map)     |
+------------------------------------------------------------------------------------+
```

---

## 2. Dashboard Architecture (`naviguard_dashboard`)

The dashboard is implemented using Python's standard `http.server.ThreadingHTTPServer` and HTML5 Canvas, requiring **zero external cloud or pip dependencies**.

### Key Modules:
- **`coordinate_converter.py`**: Handles bijective coordinate transformations between world coordinates ($X, Y$ in meters), OccupancyGrid cells ($col, row$), and HTML5 Canvas pixels ($u, v$).
- **`goal_validator.py`**: Safety gatekeeper preventing invalid goals from reaching the navigation stack (bounds checking, occupancy checks, system readiness).
- **`state_cache.py`**: Thread-safe telemetry store maintaining rolling rates, pose history, confidence breakdown, and operational event logs.
- **`web_server.py`**: HTTP REST API serving static assets, JPEG camera streams, PNG map renders, and operator command endpoints.
- **`dashboard_node.py`**: Central ROS 2 node subscribing to all 10 subsystems and publishing strictly to `/goal_pose` (never to `/cmd_vel`).

---

## 3. ROS Topics & Service Inventory

| Subsystem | Topics Subscribed / Published | Message Type |
|---|---|---|
| **Raw Vision** | `/camera/image_raw` | `sensor_msgs/msg/Image` |
| **Perception** | `/perception/debug_image` | `sensor_msgs/msg/Image` |
| **Visual Odometry** | `/visual_odometry/debug_image`<br>`/visual_odometry/telemetry` | `sensor_msgs/msg/Image`<br>`diagnostic_msgs/msg/DiagnosticArray` |
| **Sensors & Fusion** | `/imu`<br>`/odom`<br>`/state_estimation/odom` | `sensor_msgs/msg/Imu`<br>`nav_msgs/msg/Odometry`<br>`nav_msgs/msg/Odometry` |
| **SLAM & Map** | `/slam/pose`<br>`/slam/map`<br>`/slam/trajectory` | `geometry_msgs/msg/PoseWithCovarianceStamped`<br>`nav_msgs/msg/OccupancyGrid`<br>`nav_msgs/msg/Path` |
| **Confidence** | `/naviguard/decision`<br>`/naviguard/diagnostics` | `std_msgs/msg/String` (JSON)<br>`diagnostic_msgs/msg/DiagnosticArray` |
| **Recovery** | `/recovery/state`<br>`/recovery/diagnostics` | `std_msgs/msg/String` (JSON)<br>`diagnostic_msgs/msg/DiagnosticArray` |
| **Navigation** | `/navigation/state`<br>`/navigation/path`<br>`/goal_pose` | `std_msgs/msg/String` (JSON)<br>`nav_msgs/msg/Path`<br>`geometry_msgs/msg/PoseStamped` |

---

## 4. Map Coordinate Conversion Mathematics

Given an OccupancyGrid with origin $(x_{\text{origin}}, y_{\text{origin}})$, resolution $r$, dimensions $(W, H)$, and an HTML5 canvas of dimensions $(C_W, C_H)$:

### Canvas Click $(u, v) \to$ World $(x, y)$:
$$\text{col} = \left\lfloor u \cdot \frac{W}{C_W} \right\rfloor$$
$$\text{row} = \left\lfloor (C_H - 1 - v) \cdot \frac{H}{C_H} \right\rfloor$$
$$x = x_{\text{origin}} + (\text{col} + 0.5) \cdot r$$
$$y = y_{\text{origin}} + (\text{row} + 0.5) \cdot r$$

### World $(x, y) \to$ Canvas $(u, v)$:
$$\text{col} = \frac{x - x_{\text{origin}}}{r}, \quad \text{row} = \frac{y - y_{\text{origin}}}{r}$$
$$u = \text{col} \cdot \frac{C_W}{W}, \quad v = (H - 1 - \text{row}) \cdot \frac{C_H}{H}$$

This accounts for the vertical inversion between ROS (bottom-to-top) and browser canvas (top-to-bottom).

---

## 5. One-Command Master Startup

```bash
cd ~/naviguard_ws
source /opt/ros/jazzy/setup.bash
source ~/naviguard_ws/install/setup.bash

./scripts/run_naviguard.sh
```

### Readiness Check Progression:
```
[1/10] Gazebo Simulation ............ READY
[2/10] Camera & IMU Sensors ......... READY
[3/10] Vision Perception ............ READY
[4/10] Visual Odometry .............. READY
[5/10] State Estimation EKF ......... READY
[6/10] Visual SLAM & Map ............ READY
[7/10] Confidence Engine ............ READY
[8/10] Autonomous Recovery .......... READY
[9/10] Mission Navigation ........... READY
[10/10] Operator Dashboard .......... READY

✔ NAVIGUARD AUTONOMOUS STACK FULLY OPERATIONAL
Dashboard Web URL: http://localhost:8080
```

---

## 6. Operator Demonstration Flow

1. **Observe Initial Nominal State:**
   - Top status bar displays `SYSTEM: ONLINE`, `MISSION: IDLE`, `DECISION: CONTINUE`, `CONF: 0.90+`.
   - Camera panel renders live stream from Gazebo Harmonic.
   - Interactive map displays robot at $(0.0, 0.0)$ facing $+X$ along dirt trail.
2. **Dispatch Autonomous Goal:**
   - Click **Set Destination**.
   - Click a forward point on the dirt trail (e.g. $X = 3.0, Y = 0.0$).
   - Goal validator confirms: `GOAL VALID: Target is reachable and within traversable space.`
   - Click **Start Navigation**.
3. **Autonomous Navigation & Replanning:**
   - Navigation node computes A* global plan and publishes `/navigation/path`.
   - Robot starts forward motion; VO telemetry reflects optical flow and positive forward velocity.
   - Local costmap identifies roadside obstacle (fallen logs / rock cairn) and replans path dynamically.
4. **Controlled Fault Injection:**
   - Operator clicks **Trigger Fault** in the dashboard footer.
   - `/recovery/trigger_manual_recovery` service executes.
   - Confidence drops; state transitions: `CONTINUE` $\to$ `VERIFY` $\to$ `SAFE_STOP` $\to$ `RECOVER`.
   - Recovery controller performs rotational visual reacquisition / backtrack motion.
   - Once tracking is regained, recovery yields control and mission resumes automatically.
5. **Mission Completion:**
   - Robot reaches spatial tolerance of goal ($\le 0.20\text{ m}$).
   - Status transitions to `GOAL_REACHED`.

---

## 7. Clean Shutdown

Pressing `Ctrl+C` in the running terminal triggers `./scripts/stop_naviguard.sh`:
- Sends `SIGINT` to main launch process.
- Terminates ros_gz_bridge, Gazebo Harmonic, and all 10 Python autonomy nodes.
- Zero lingering background processes or CPU leaks.
