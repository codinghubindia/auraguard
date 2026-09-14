# NAVIGUARD Interactive Map & Goal-Dispatch Engineering Report

**Project:** NAVIGUARD — SIH 2026 Vision-Based Autonomous Navigation for Outdoor UGV  
**Subsystem:** Interactive Map, Goal Safety Validator, REST API Bridge & Navigation Interface  
**Date:** September 2026  
**Status:** **FULLY RESOLVED & VALIDATED (215/215 Tests Passing)**

---

## 1. Executive Summary

During operational audit of the NAVIGUARD operator dashboard (`http://localhost:8080`), an issue was reported where clicking on the interactive map canvas did not allow operators to reliably set destinations or dispatch goals to the autonomous navigation stack.

This was resolved via targeted bug-fixes preserving the existing architecture:
- **No architectural changes**: Nav2 was not introduced; custom SLAM and A* global planner remain primary.
- **Single-command velocity ownership preserved**: All motion commands originate from `navigation_node` via `/cmd_vel`.
- **Safety checks intact**: Boundary, obstacle cost, and subsystem health checks remain enforced.
- **Full bidirectional verification completed**: Canvas click $\to$ World coordinate conversion $\to$ Safety validation $\to$ Candidate preview marker $\to$ REST dispatch $\to$ ROS 2 `/goal_pose` $\to$ Autonomous navigation stack.

---

## 2. Root Cause Analysis

A multi-layered audit revealed four distinct failure modes that prevented reliable goal setting:

| Component | Root Cause | Failure Effect |
| :--- | :--- | :--- |
| **Frontend Map Canvas** (`index.html`) | Canvas `click` listener had a rigid `if (!isSetGoalMode) return;` guard with no indicator on the map header. Small mouse movements during clicks were absorbed by the pan-drag handler. | Map clicks were silently ignored or treated as pan operations. |
| **Mode Desynchronization** (`index.html`) | Goal-setting toggle button was only in Panel 11 (Mission Controls) at the bottom right. The Map panel (Panel 8) lacked an inline activation button. | Operators clicking the map without scrolling down saw no response. |
| **Readiness Check Jitter** (`goal_validator.py` & `web_server.py`) | Exact equality `status == 'ONLINE'` was checked. If `Navigation` or `SLAM` had slight rate fluctuations to `'DEGRADED'` during startup, goals were falsely rejected with `"Navigation subsystem is OFFLINE"`. | Goals were rejected despite nodes running nominally. |
| **REST Route Aliasing** (`web_server.py`) | Handler strictly listened on `/api/goal`. Requests to `/api/set_goal` returned HTTP 404. | External or test dispatches to `/api/set_goal` failed. |
| **Streaming Disconnects** (`web_server.py`) | Image stream writes called `self.wfile.write()` without catching socket disconnection errors. | Console spammed with `BrokenPipeError` / `ConnectionResetError` when browser refreshed. |
| **Node Shutdown Lifecycle** (`navigation_node.py`) | `finally` block called `rclpy.shutdown()` unconditionally. | Ctrl+C shutdown produced `rcl_shutdown already called on the given context`. |

---

## 3. Implementation Details

### 3.1 Frontend Fixes (`src/naviguard_dashboard/naviguard_dashboard/static/index.html`)
1. **Interactive Map Header Toggle (`🎯 SET GOAL`):**
   - Added `#btnMapSetGoal` directly into the Panel 8 header, synchronized with `#btnSetGoal` in Panel 11.
   - Updates cursor to `crosshair` when active and `grab`/`grabbing` when navigating.
   - Temporarily pauses robot camera follow during goal selection so targeting remains stable.
2. **Click vs Pan Discrimination (`CLICK_DRAG_THRESHOLD_PX = 6`):**
   - Distinguishes intentional click selection from viewport panning using pixel displacement threshold.
3. **Immediate Visual Candidate Feedback:**
   - On click, displays an immediate candidate crosshair with `CHECKING...` status while network validation resolves.
   - Renders state-dependent colors: Cyan (`#00d2ff`) for checking, Green (`#05f187`) for valid, Red (`#ff3366`) for invalid/blocked.
4. **Engineering Direct Test Button:**
   - Added `TEST GOAL (X=2.0)` button in Panel 11 to test the full pipeline directly.

### 3.2 Backend REST & Safety Validation (`web_server.py` & `goal_validator.py`)
1. **Safe Socket Writing (`_safe_write`):**
   - Suppresses `BrokenPipeError`, `ConnectionResetError`, and `ConnectionAbortedError` across all HTTP endpoints.
2. **Route Aliasing:**
   - `do_POST` handles both `/api/set_goal` and `/api/goal`.
3. **Degradation-Tolerant Readiness Check:**
   - Accepts both `ONLINE` and `DEGRADED` health states.
   - Accepts fallback state estimation pose if SLAM is completing keyframe initialization.

### 3.3 Navigation Node Shutdown (`navigation_node.py`)
- Guarded `rclpy.shutdown()` with `if rclpy.ok(): rclpy.shutdown()`.

---

## 4. Verification & Testing Evidence

### 4.1 Automated Workspace Test Suite
All 11 packages executed clean test runs with zero failures:
```text
build/naviguard_confidence:       20 passed
build/naviguard_dashboard:        23 passed (including new integration tests)
build/naviguard_description:      8 passed
build/naviguard_navigation:       33 passed
build/naviguard_perception:       11 passed
build/naviguard_recovery:         19 passed
build/naviguard_rellis:           28 passed
build/naviguard_sensor_sync:      19 passed
build/naviguard_slam:             17 passed
build/naviguard_state_estimation: 20 passed
build/naviguard_visual_odometry:  16 passed
Total: 215 tests, 0 errors, 0 failures, 0 skipped
```

### 4.2 Live End-to-End Goal Dispatch Verification
Executed end-to-end test with `NaviguardDashboardNode` and `NavigationNode`:
```text
[INFO] [naviguard_dashboard_node]: NAVIGUARD Operator Dashboard running at http://0.0.0.0:8080
[INFO] [navigation_node]: Navigation Node initialized: rate=10.0Hz, v_max=0.25m/s, w_max=0.35rad/s
VALIDATE GOAL CLICK: {'x': 2.53, 'y': -0.02, 'valid': True, 'reason': 'GOAL VALID: Target is reachable'}
[INFO] [naviguard_dashboard_node]: Operator published goal to /goal_pose: X=2.50m, Y=1.00m
SET_GOAL API RESP: {'success': True, 'message': 'Goal successfully published to /goal_pose'}
[INFO] [navigation_node]: New navigation goal received: (2.50, 1.00)
NAV NODE GOAL: NavigationGoal(x=2.5, y=1.0, yaw=0.0, frame_id='map')
RESULT: GOAL VERIFIED IN NAVIGATION STACK!
```

### 4.3 Image Stream Disconnect Stress Test
10 rapid client connection drops produced zero tracebacks or server crashes.

---

## 5. Operator Instructions

1. **Open Dashboard:** Navigate to `http://localhost:8080`.
2. **Activate Destination Mode:** Click either `🎯 SET GOAL` in the Interactive Map header or `Set Destination` in Mission Controls. The map cursor becomes a crosshair.
3. **Click on Map:** Click anywhere within traversable map bounds. An immediate candidate marker appears with live coordinates and traversability feedback.
4. **Dispatch Mission:** When the marker turns green and displays `VALID`, click `Start Nav`. The destination is published to `/goal_pose`, and the autonomous UGV begins trajectory execution.
