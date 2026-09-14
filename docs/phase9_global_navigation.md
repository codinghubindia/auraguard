# NAVIGUARD Phase 9 — Mission Management & Global Navigation Report

**Project**: NAVIGUARD — Confidence-Aware Closed-Loop Navigation & Autonomous Recovery for an Outdoor UGV  
**Milestone**: Phase 9 (Mission Management, A* Global Planning, and Path Following)  
**Date**: September 2026  
**Engineer**: NAVIGUARD Mission & Global Navigation Lead  
**Workspace**: `~/naviguard_ws`  
**Environment**: Ubuntu 24.04 Noble, ROS 2 Jazzy, Gazebo Harmonic, WSL2  

---

## 1. Objective
Phase 9 implements the autonomous START $\to$ DESTINATION navigation layer for the NAVIGUARD outdoor UGV. Prior to this milestone, the robot was limited to open-loop demo motions and reactive backtracking recoveries. Phase 9 introduces a complete, self-contained, and mathematically sound navigation package (`naviguard_navigation`) capable of:
1. Managing mission state transitions (`IDLE` through `GOAL_REACHED` / `MISSION_FAILED`).
2. Expressing and validating global destinations in the `map` coordinate frame.
3. Transforming 2D occupancy grids with robot footprint inflation and clearance cost fields.
4. Planning optimal, collision-free paths using an 8-connected grid A* planner.
5. Smoothing paths via collision-verified line-of-sight shortcutting.
6. Generating discrete waypoints with assigned orientations.
7. Tracking waypoints closed-loop using pure pursuit with lookahead horizon and angular rate limiting.
8. Detecting spatial arrival and enforcing continuous settlement dwell criteria.
9. Dynamically monitoring path validity and cross-track deviation to trigger debounced replanning.
10. Coordinating cleanly with Phase 7 Confidence and Phase 8 Autonomous Recovery with strict mutual exclusion on `/cmd_vel`.

---

## 2. Architecture & Dataflow

```
                     ┌───────────────────────────┐
                     │    /goal_pose (ROS 2)     │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │      GoalManager      │
                       └───────────┬───────────┘
                                   │
  /slam/map (OccupancyGrid)        ▼
  ───────────────► ┌───────────────────────────────┐
                   │    NavigationOccupancyGrid    │
                   │ (Footprint Inflation + Cost)  │
                   └───────────────┬───────────────┘
                                   │
  /slam/pose (PoseWithCovariance)  ▼
  ───────────────► ┌───────────────────────────────┐
                   │      GlobalPlannerAStar       │
                   │    (8-Connected Grid A*)      │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │         PathSmoother          │
                   │ (Collision-Safe Shortcutting) │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │       WaypointGenerator       │
                   │      (Headings & Spacing)     │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │          PathFollower         │
                   │      (Pure Pursuit Lookahead) │
                   └───────────────┬───────────────┘
                                   │
  /naviguard/decision (Phase 7)    │
  ─────────────────────────────┐   │
  /recovery/state (Phase 8)    ▼   ▼
  ───────────────────────────► ┌───────────────────────┐
                               │   CmdVel Arbitration  ├────► /cmd_vel (to ros_gz_bridge)
                               └───────────────────────┘
```

---

## 3. Mission State Machine

Implemented in [`mission_manager.py`](file:///home/maxx/naviguard_ws/src/naviguard_navigation/naviguard_navigation/mission_manager.py):

* **`IDLE` (0)**: Initial quiescent state. Zero velocity commanded.
* **`GOAL_SET` (1)**: Valid destination received from `/goal_pose` or `/navigation/set_goal`. Awaiting valid map and pose.
* **`PLANNING` (2)**: A* global planner actively computing collision-free path.
* **`NAVIGATING` (3)**: Path follower actively driving the UGV toward waypoints.
* **`REPLANNING` (4)**: Triggered when upcoming path is blocked by newly mapped obstacles or cross-track deviation exceeds tolerance.
* **`RECOVERY_WAIT` (5)**: Navigation yields 100% control of `/cmd_vel` to Phase 8 Recovery (`recovery_node`).
* **`GOAL_REACHED` (6)**: Robot has settled within position and orientation tolerances for the required dwell time.
* **`MISSION_FAILED` (7)**: Planning retry budget exhausted or Phase 8 recovery ended in `FAILED_SAFE`.

---

## 4. Goal Interface
* **Topic**: `/goal_pose` and `/navigation/set_goal` (`geometry_msgs/msg/PoseStamped`) in coordinate frame `map`.
* **Cancellation**: Service `/navigation/cancel_goal` (`std_srvs/srv/Trigger`) cleanly aborts active navigation and returns the robot to `IDLE`.
* **Script**: Dedicated helper [`scripts/send_navigation_goal.sh`](file:///home/maxx/naviguard_ws/scripts/send_navigation_goal.sh) handles subscriber discovery and publishes goals directly from the CLI.

---

## 5. Occupancy-Grid Processing & Footprint Inflation

Implemented in [`occupancy_grid.py`](file:///home/maxx/naviguard_ws/src/naviguard_navigation/naviguard_navigation/occupancy_grid.py):
* **Resolution**: $0.05\text{ m/cell}$ matching SLAM.
* **Robot Footprint**:
  * Chassis: $0.50\text{ m} \times 0.32\text{ m}$, track width $0.48\text{ m}$.
  * Circumscribed radius: $R_{circ} \approx 0.35\text{ m}$.
  * Configured safety margin: $R_{margin} = 0.15\text{ m}$.
  * **Lethal Inflation Radius**: $R_{inflate} = R_{circ} + R_{margin} = 0.50\text{ m}$ ($10\text{ cells}$).
* **Proximity Cost Field**: Smooth linear decay between $R_{inflate}$ ($0.50\text{ m}$) and $R_{prox}$ ($1.00\text{ m}$), pushing paths toward open, safer corridors.
* **Unknown Space Policy**: Configurable (`allow_unknown: true`, `unknown_cost_penalty: 5.0`). Cells marked `-1` are traversable with cost penalties, allowing exploration toward unmapped goals while strictly avoiding confirmed obstacles.

---

## 6. A* Implementation & Multi-Factor Cost Model

Implemented in [`global_planner.py`](file:///home/maxx/naviguard_ws/src/naviguard_navigation/naviguard_navigation/global_planner.py):
* **Neighborhood**: 8-connected grid with diagonal safety validation (prevents cutting through diagonal obstacle corners).
* **Search Mechanics**: Min-heap priority queue with Euclidean heuristic $h(n) = \sqrt{\Delta x^2 + \Delta y^2}$.
* **Cost Function**:
  $$g(n_{next}) = g(n) + \text{step\_dist} \times \left(1.0 + w_{prox} \cdot c_{proximity}\right) + w_{turn} \cdot \Delta \theta$$
  * $\text{step\_dist}$: $0.05\text{ m}$ (orthogonal) or $0.0707\text{ m}$ (diagonal).
  * $w_{prox} \cdot c_{proximity}$: Penalty derived from the obstacle clearance distance transform.
  * $w_{turn} \cdot \Delta \theta$: Direction change penalty ($0.5 \times \text{resolution}$) penalizing unnecessary zigzagging.

---

## 7. Path Smoothing & Waypoint Generation

* **Path Smoother** ([`path_smoother.py`](file:///home/maxx/naviguard_ws/src/naviguard_navigation/naviguard_navigation/path_smoother.py)): Greedy line-of-sight string-pulling. Uses Bresenham raycasting across the inflated cost grid. **Strict Safety Invariant**: Every shortcut segment is validated against the inflated grid; if any point intersects an obstacle, the shortcut is rejected and conservative routing is preserved.
* **Waypoint Generator** ([`waypoint_generator.py`](file:///home/maxx/naviguard_ws/src/naviguard_navigation/naviguard_navigation/waypoint_generator.py)): Samples the continuous path at regular intervals ($0.35\text{ m}$ target spacing), preserving critical corners and assigning tangent headings $\theta = \text{atan2}(\Delta y, \Delta x)$.

---

## 8. Path Following Controller

Implemented in [`path_follower.py`](file:///home/maxx/naviguard_ws/src/naviguard_navigation/naviguard_navigation/path_follower.py):
* **Pure Pursuit with Lookahead**: Dynamic lookahead horizon $L = 0.40\text{ m}$.
* **Heading & Cross-Track Feedback**:
  $$\omega_z = k_h \cdot e_\theta - k_{ct} \cdot e_{lat}$$
  Clamped to $[-0.35, +0.35]\text{ rad/s}$.
* **Adaptive Linear Velocity**:
  * Large heading error ($|e_\theta| > 0.75\text{ rad} \approx 43^\circ$): in-place turn ($v_x = 0.0$) to align before driving forward.
  * Aligned tracking: $v_x = v_{max} \cdot \cos(e_\theta)$.
  * Final approach ($d_{goal} < 1.0\text{ m}$): smooth deceleration to docking velocity ($0.05\text{ m/s}$).

---

## 9. Goal Reached Verification

Implemented in [`goal_checker.py`](file:///home/maxx/naviguard_ws/src/naviguard_navigation/naviguard_navigation/goal_checker.py):
* **Spatial Tolerance**: $d \le 0.25\text{ m}$.
* **Orientation Tolerance**: $|\Delta \theta| \le 0.30\text{ rad} \approx 17^\circ$.
* **Continuous Dwell**: Robot must remain continuously within tolerances for $\ge 1.0\text{ second}$ to prevent premature success declarations caused by sensor noise.

---

## 10. Replanning vs. Recovery Coordination

| Condition | Responsible Layer | Action |
| :--- | :--- | :--- |
| **New obstacle on path** | Phase 9 `Replanner` | Replan A* path around obstacle (`REPLANNING` $\to$ `NAVIGATING`) |
| **Cross-track error $> 0.60\text{ m}$** | Phase 9 `Replanner` | Replan from current pose to goal |
| **Confidence: `VERIFY`** | Phase 9 `PathFollower` | Reduce linear velocity by 50% ($v_{max} = 0.12\text{ m/s}$) |
| **Confidence: `RECOVER`** | Phase 7 / 8 `Recovery` | Navigation yields `/cmd_vel` immediately (`RECOVERY_WAIT`) |
| **Recovery Active** | Phase 8 `recovery_node` | Recovery executes backtrack; Navigation is silent on `/cmd_vel` |
| **Recovery Succeeded** | Phase 9 `MissionManager` | Transitions `RECOVERY_WAIT` $\to$ `REPLANNING` $\to$ `NAVIGATING` |
| **Recovery Failed Safe** | Phase 8 $\to$ Phase 9 | Navigation enters `MISSION_FAILED`; zero velocity enforced |

---

## 11. Command Velocity (`/cmd_vel`) Ownership Rules

To prevent command contention and oscillation:
1. **`NAVIGATING`**: `navigation_node` exclusively publishes to `/cmd_vel`.
2. **`IDLE` / `GOAL_REACHED` / `MISSION_FAILED`**: `navigation_node` publishes one zero velocity command, then halts.
3. **`RECOVERY_WAIT` / `REPLANNING`**: `navigation_node` **never publishes to `/cmd_vel`**, yielding 100% control to `recovery_node`.

---

## 12. RViz Visualization Profile

Available in [`navigation_view.rviz`](file:///home/maxx/naviguard_ws/src/naviguard_navigation/config/navigation_view.rviz):
* Cyan Path: Global planned path (`/navigation/path`).
* Orange Arrows: Discrete waypoints (`/navigation/waypoints`).
* Green Cylinder: Target destination marker (`/navigation/markers`).
* Orange Sphere: Lookahead tracking target (`/navigation/markers`).
* Green Arrow: Authoritative robot pose from SLAM (`/slam/pose`).
* Red Line: Historical trajectory from SLAM (`/slam/trajectory`).
* Blue Halo: Confidence decision state marker (`/naviguard/decision_marker`).
* Checkpoints: Recovery breadcrumbs and safe return targets (`/recovery/visualization`).

---

## 13. Test Results & Verification

### Unit & Scenario Test Suite (30 Tests in `naviguard_navigation`)
All tests executed deterministically:
* `test_occupancy_grid.py`: Coordinate conversions, obstacle inflation, unknown space policy.
* `test_global_planner_astar.py`: Basic path planning, wall obstacle routing, start==goal, unreachable goal.
* `test_path_smoother.py`: String-pulling shortcutting, obstacle preservation.
* `test_waypoint_generator.py`: Uniform spacing, heading tangent generation, goal preservation.
* `test_goal_checker.py`: Spatial tolerances, orientation verification, continuous dwell timer.
* `test_replanner.py`: Dynamic obstacle detection on active path, excessive deviation detection.
* `test_mission_manager.py`: Complete lifecycle state transitions, retry budget exhaustion.
* `test_navigation_integration.py`: Node lifecycle, goal setting & canceling service, recovery yielding.
* `test_navigation_scenarios.py`: Full scenarios Tests A through I.

### Workspace Regression Summary
```text
build/naviguard_confidence:       20 tests, 0 errors, 0 failures, 0 skipped
build/naviguard_description:       8 tests, 0 errors, 0 failures, 0 skipped
build/naviguard_navigation:       30 tests, 0 errors, 0 failures, 0 skipped  [NEW]
build/naviguard_perception:        9 tests, 0 errors, 0 failures, 0 skipped
build/naviguard_recovery:         19 tests, 0 errors, 0 failures, 0 skipped
build/naviguard_sensor_sync:      19 tests, 0 errors, 0 failures, 0 skipped
build/naviguard_slam:             15 tests, 0 errors, 0 failures, 0 skipped
build/naviguard_state_estimation: 20 tests, 0 errors, 0 failures, 0 skipped
build/naviguard_visual_odometry:  15 tests, 0 errors, 0 failures, 0 skipped

TOTAL: 156 tests, 0 errors, 0 failures, 0 skipped
```

---

## 14. Live Simulation Verification Evidence

During live execution of `naviguard_demo.launch.py headless:=true start_demo_motion:=false start_navigation:=true`:

1. **Active Nodes**: 10/10 core nodes running concurrently, including `navigation_node`.
2. **Goal Reception**: Goal $(1.50, 0.00)$ was published to `/goal_pose`.
3. **Planning & Trajectory**:
   * A* planner computed collision-free path (`29 waypoints`).
   * `Mission State: NAVIGATING [Ownership: NAVIGATING_ACTIVE]`.
   * Robot tracked waypoints smoothly ($v_x = 0.25\text{ m/s}$, position moved from $0.00\text{ m}$ to $0.59\text{ m}$).
4. **Fault Injection & Recovery Coordination**:
   * Manual fault injected via `inject_fault.sh trigger`.
   * `navigation_node` immediately yielded `/cmd_vel` (`RECOVERY_WAIT`).
   * `recovery_node` executed `SHORT_BACKTRACK` reverse maneuver without command interference.
   * Recovery budget reset via `inject_fault.sh reset`, allowing navigation to resume cleanly toward the original goal.

---

## 15. Known Limitations
1. **Monocular Scale Dependency**: Without external depth sensors, SLAM landmark triangulation scale is grounded by wheel odometry and IMU integration.
2. **Static Inflation Radius**: The inflation radius is fixed at $0.50\text{ m}$. In extremely narrow outdoor passages, a dynamically adaptive inflation cost may be advantageous.

---

## 16. Exact Phase 10 Recommendations
1. **RELLIS-3D Dataset Integration**: Introduce rugged outdoor off-road terrains, vegetation, mud, and uneven ground geometries derived from RELLIS-3D.
2. **Traversability Costmapping**: Integrate terrain classification scores from perception into the occupancy grid cost layer (e.g. grass vs. rock vs. mud penalties).
3. **Advanced Path Tracking**: Evaluate Model Predictive Path-Integral (MPPI) control for dynamic off-road tracking on slip-prone terrain.
