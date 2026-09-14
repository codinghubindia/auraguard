# NAVIGUARD Phase 10 — Ground Truth Isolation & Anti-Cheating Architectural Audit

**Milestone**: Phase 10 (RELLIS-3D Integration & Evaluation)  
**Auditor**: NAVIGUARD Systems & Evaluation Engineer  
**Scope**: All 10 workspace packages (`src/`)  

---

## 1. Ground Truth Isolation Architecture

To ensure strict engineering integrity, NAVIGUARD enforces an absolute barrier between:
1. **The Autonomous Navigation Runtime Stack** (Perception, VO, State Estimation, SLAM, Confidence, Recovery, Navigation).
2. **The Evaluation & Benchmarking Pipeline** (Dataset ground truth, error metrics, trajectory evaluation).

```
  ┌────────────────────────────────────────────────────────┐
  │              RELLIS-3D / GAZEBO SOURCE                 │
  └───────────┬────────────────────────────────┬───────────┘
              │                                │
    Camera Images + Intrinsics                 │ Ground Truth Poses + GPS/INS
              │                                │
              ▼                                ▼
  ┌───────────────────────────────┐ ┌──────────────────────────────────────┐
  │      AUTONOMOUS RUNTIME       │ │          GROUND TRUTH GUARD          │
  │    (naviguard_navigation)     │ │        (Evaluation Only Layer)       │
  │                               │ │                                      │
  │  • Shi-Tomasi + LK VO         │ │  • ATE RMSE Calculation              │
  │  • EKF State Estimation       │ │  • RPE Calculation                   │
  │  • Feature Visual SLAM        │ │  • Trajectory Alignment              │
  │  • Multi-factor Confidence    │ │  • Ground Truth RViz Markers         │
  │  • Autonomous Recovery FSM    │ │                                      │
  │  • A* Global Path Planning    │ │  [STRICT ISOLATION: NEVER WRITES TO] │
  │  • Pure Pursuit Control       │ │  [    /slam/pose, /cmd_vel, /odom  ] │
  └───────────────┬───────────────┘ └──────────────────────────────────────┘
                  │
                  ▼
              /cmd_vel ──► UGV Motors
```

---

## 2. Codebase Audit Matrix

| Source / Symbol | Location / Package | Runtime Use | Allowed or Forbidden | Audit Result | Evidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`poses.txt` (RELLIS GT)** | `naviguard_rellis` | Offline ATE/RPE error calculation | **ALLOWED** in Evaluation only | **VERIFIED CLEAN** | Loaded exclusively in `evaluation_recorder.py` and `pose_loader.py`. Never connected to `/slam/pose`. |
| **`gps_ins.txt` (GNSS/INS)** | `naviguard_rellis` | Benchmark reference frame | **ALLOWED** in Evaluation only | **VERIFIED CLEAN** | No GNSS subscriber or topic bridge exists in the robot navigation stack. |
| **`sensor_msgs/NavSatFix`** | Workspace-wide | None | **FORBIDDEN** in Navigation | **PASS** (Zero occurrences found in codebase). | Zero imports in `src/`. |
| **Gazebo `/model_states`** | Simulation bridge | None | **FORBIDDEN** in Navigation | **PASS** | Bridge configuration (`bridge.yaml`) contains only standard sensor topics (`/odom`, `/imu`, `/camera/*`). No model state bridge. |
| **Gazebo `/link_states`** | Simulation bridge | None | **FORBIDDEN** in Navigation | **PASS** | Excluded from bridge configuration. |
| **Ground-Truth Odometry Teleportation** | Workspace-wide | None | **FORBIDDEN** | **PASS** | Vehicle motion driven strictly by wheel torque and `/cmd_vel`. |
| **Pre-recorded Path Following** | `naviguard_navigation` | Real-time A* computation | **FORBIDDEN** if precomputed | **PASS** | Paths generated dynamically on live occupancy grid by `GlobalPlannerAStar`. |

---

## 3. Enforcement: `GroundTruthGuard` Component
The class `GroundTruthGuard` in `naviguard_rellis` acts as a programmatic barrier:
* Rejects any attempt to publish ground-truth data onto authoritative robot topics (`/slam/pose`, `/state_estimation/odom`, `/odom`, `/cmd_vel`).
* Strips reference data before dispatching telemetry.
* Emits a fatal assertion error if any runtime module requests ground truth for navigation decisions.
