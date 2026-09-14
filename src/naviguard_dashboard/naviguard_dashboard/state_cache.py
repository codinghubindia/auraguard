"""
Thread-safe Telemetry & State Cache for NAVIGUARD Operator Dashboard.

Maintains live snapshots of robot states, camera feeds, occupancy grid,
and system health metrics for web client consumption.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple


class StateCache:
    """Thread-safe storage for live ROS 2 telemetry and sensor data."""

    def __init__(self, max_events: int = 50, max_traj_points: int = 500):
        self._lock = threading.Lock()
        self.max_events = max_events
        self.max_traj_points = max_traj_points

        # 1. System Health & Rates
        self.system_status = "INITIALIZING"
        self.last_update_time = time.time()
        self.subsystems: Dict[str, Dict[str, Any]] = {
            "Gazebo": {"status": "OFFLINE", "rate": 0.0, "unit": "Hz"},
            "Camera": {"status": "OFFLINE", "rate": 0.0, "unit": "Hz"},
            "IMU": {"status": "OFFLINE", "rate": 0.0, "unit": "Hz"},
            "Odometry": {"status": "OFFLINE", "rate": 0.0, "unit": "Hz"},
            "Perception": {"status": "OFFLINE", "rate": 0.0, "unit": "FPS"},
            "Visual Odometry": {"status": "OFFLINE", "rate": 0.0, "unit": "FPS"},
            "State Estimation": {"status": "OFFLINE", "rate": 0.0, "unit": "Hz"},
            "SLAM": {"status": "OFFLINE", "rate": 0.0, "unit": "Hz"},
            "Confidence": {"status": "OFFLINE", "rate": 0.0, "unit": "Hz"},
            "Recovery": {"status": "OFFLINE", "rate": 0.0, "unit": "Hz"},
            "Navigation": {"status": "OFFLINE", "rate": 0.0, "unit": "Hz"},
            "Dashboard": {"status": "ONLINE", "rate": 1.0, "unit": "Hz"},
        }

        # 2. Top-Level States
        self.mission_state = "IDLE"
        self.confidence_state = "CONTINUE"
        self.recovery_state = "NORMAL"

        # 3. Multi-dimensional Confidence Breakdown
        self.confidence_scores: Dict[str, float] = {
            "overall": 1.0,
            "visual": 1.0,
            "localization": 1.0,
            "imu": 1.0,
            "wheel": 1.0,
            "temporal": 1.0,
            "cross_sensor": 1.0,
            "map": 1.0,
        }
        self.confidence_primary_reason = "NOMINAL_OPERATION"

        # 4. Recovery Subsystem Detail
        self.recovery_strategy = "None"
        self.recovery_attempts = 0
        self.recovery_max_attempts = 3
        self.recovery_dwell_sec = 0.0

        # 5. Visual Motion & VO Telemetry
        self.vo_telemetry: Dict[str, Any] = {
            "num_detected": 0,
            "num_tracked": 0,
            "num_inliers": 0,
            "tracking_ratio": 0.0,
            "median_displacement_px": 0.0,
            "geom_status": "WAITING",
            "rot_yaw_deg": 0.0,
            "scale_status": "UNKNOWN",
            "input_fps": 0.0,
            "proc_fps": 0.0,
        }

        # 6. Localization & Navigation Geometry
        self.robot_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self.trajectory: List[Dict[str, float]] = []
        self.nav_path: List[Dict[str, float]] = []
        self.active_goal: Optional[Dict[str, float]] = None
        self.nav_stats: Dict[str, Any] = {
            "dist_to_goal_m": 0.0,
            "current_waypoint_idx": 0,
            "total_waypoints": 0,
            "replan_count": 0,
            "cmd_vx": 0.0,
            "cmd_wz": 0.0,
            "path_valid": False,
        }
        self.replan_diagnostics: Dict[str, Any] = {
            "replan_reason": "None",
            "status": "NOMINAL",
            "old_path_length_m": 0.0,
            "new_path_length_m": 0.0,
            "path_difference_m": 0.0,
            "blocked_regions_count": 0,
            "dist_to_goal_m": 0.0,
            "timestamp": 0.0,
        }

        # 7. Map Metadata & Grid
        self.map_meta: Dict[str, Any] = {
            "width": 600,
            "height": 600,
            "resolution": 0.05,
            "origin_x": -15.0,
            "origin_y": -15.0,
            "has_data": False,
        }
        self.grid_data: Optional[List[int]] = None
        self.map_png_bytes: Optional[bytes] = None

        # 8. Encoded Camera Frames (JPEG bytes)
        self.raw_cam_jpeg: Optional[bytes] = None
        self.perception_jpeg: Optional[bytes] = None
        self.segmentation_jpeg: Optional[bytes] = None
        self.vo_jpeg: Optional[bytes] = None
        self.chase_jpeg: Optional[bytes] = None


        # 9. Rolling Event Log
        self.events: List[Dict[str, Any]] = []
        self.add_event("SYSTEM", "NAVIGUARD Operator Dashboard initialized.")

    def add_event(self, category: str, text: str) -> None:
        """Add an event to rolling log."""
        timestamp_str = time.strftime("%H:%M:%S")
        entry = {"time": timestamp_str, "category": category, "text": text}
        with self._lock:
            self.events.insert(0, entry)
            if len(self.events) > self.max_events:
                self.events.pop()

    def update_subsystem_rate(self, name: str, rate: float, is_online: bool = True) -> None:
        with self._lock:
            if name in self.subsystems:
                self.subsystems[name]["rate"] = round(rate, 1)
                self.subsystems[name]["status"] = "ONLINE" if is_online else "OFFLINE"

    def update_robot_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        with self._lock:
            self.robot_pose = {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3), "yaw": round(yaw, 3)}
            # Append to trajectory if moved significantly
            if not self.trajectory:
                self.trajectory.append({"x": x, "y": y})
            else:
                last = self.trajectory[-1]
                dx = x - last["x"]
                dy = y - last["y"]
                if (dx * dx + dy * dy) > 0.01:  # 10 cm movement
                    self.trajectory.append({"x": round(x, 2), "y": round(y, 2)})
                    if len(self.trajectory) > self.max_traj_points:
                        self.trajectory.pop(0)

    def set_nav_path(self, points: List[Tuple[float, float]]) -> None:
        with self._lock:
            self.nav_path = [{"x": round(p[0], 2), "y": round(p[1], 2)} for p in points]

    def set_active_goal(self, goal: Optional[Dict[str, float]]) -> None:
        with self._lock:
            self.active_goal = goal

    def get_snapshot(self) -> Dict[str, Any]:
        """Return full JSON-serializable snapshot of dashboard state."""
        with self._lock:
            # Determine overall system health
            online_nodes = sum(1 for s in self.subsystems.values() if s["status"] == "ONLINE")
            if online_nodes >= 8:
                sys_status = "ONLINE"
            elif online_nodes >= 4:
                sys_status = "DEGRADED"
            else:
                sys_status = "OFFLINE"

            return {
                "system_status": sys_status,
                "mission_state": self.mission_state,
                "confidence_state": self.confidence_state,
                "recovery_state": self.recovery_state,
                "confidence_scores": dict(self.confidence_scores),
                "confidence_primary_reason": self.confidence_primary_reason,
                "recovery_strategy": self.recovery_strategy,
                "recovery_attempts": self.recovery_attempts,
                "recovery_max_attempts": self.recovery_max_attempts,
                "recovery_dwell_sec": round(self.recovery_dwell_sec, 1),
                "vo_telemetry": dict(self.vo_telemetry),
                "robot_pose": dict(self.robot_pose),
                "trajectory": list(self.trajectory),
                "nav_path": list(self.nav_path),
                "active_goal": dict(self.active_goal) if self.active_goal else None,
                "nav_stats": dict(self.nav_stats),
                "replan_diagnostics": dict(self.replan_diagnostics),
                "map_meta": dict(self.map_meta),
                "subsystems": dict(self.subsystems),
                "events": list(self.events),
                "timestamp": time.time(),
            }

    def set_jpeg_frame(self, frame_type: str, jpeg_bytes: bytes) -> None:
        with self._lock:
            if frame_type == "raw":
                self.raw_cam_jpeg = jpeg_bytes
            elif frame_type == "perception":
                self.perception_jpeg = jpeg_bytes
            elif frame_type == "segmentation":
                self.segmentation_jpeg = jpeg_bytes
            elif frame_type == "vo":
                self.vo_jpeg = jpeg_bytes
            elif frame_type == "chase":
                self.chase_jpeg = jpeg_bytes

    def get_jpeg_frame(self, frame_type: str) -> Optional[bytes]:
        with self._lock:
            if frame_type == "raw":
                return self.raw_cam_jpeg
            elif frame_type == "perception":
                return self.perception_jpeg
            elif frame_type == "segmentation":
                return self.segmentation_jpeg
            elif frame_type == "vo":
                return self.vo_jpeg
            elif frame_type == "chase":
                return self.chase_jpeg
            return None

