"""
Thread-safe Telemetry & State Cache for NAVIGUARD Operator Dashboard.

Maintains live snapshots of robot states, camera feeds, occupancy grid,
and system health metrics for web client consumption.
Implements bounded latest-frame buffers (size = 1) for visual streams.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Tuple


class StreamBuffer:
    """Bounded latest-frame buffer (size = 1) for live operator visual streams."""

    def __init__(self, name: str, target_fps: float = 12.0) -> None:
        self.name = name
        self.target_fps = target_fps
        self.min_period = 1.0 / max(1.0, target_fps)
        self.jpeg_bytes: Optional[bytes] = None
        self.timestamp: float = 0.0
        self.frame_seq: int = 0
        self.source_fps: float = 0.0
        self.display_fps: float = 0.0
        self.encode_latency_ms: float = 0.0
        self.dropped_frames: int = 0
        self.total_frames_received: int = 0
        self._last_display_time: float = 0.0
        self._recent_display_times: List[float] = []

    def update_frame(
        self,
        jpeg_bytes: bytes,
        timestamp: float,
        encode_latency_ms: float = 0.0,
        source_fps: float = 0.0,
    ) -> None:
        """Replace buffer with newest frame (buffer size = 1, overwriting stale)."""
        self.jpeg_bytes = jpeg_bytes
        self.timestamp = timestamp
        self.frame_seq += 1
        self.total_frames_received += 1
        self.encode_latency_ms = round(encode_latency_ms, 2)
        if source_fps > 0.0:
            self.source_fps = round(source_fps, 1)

    def consume_latest(self) -> Tuple[Optional[bytes], float, float, str]:
        """Consume the newest frame, updating display FPS and calculating frame age."""
        now = time.time()
        age_ms = max(0.0, (now - self.timestamp) * 1000.0) if self.timestamp > 0.0 else 0.0

        # Track display FPS
        self._recent_display_times.append(now)
        if len(self._recent_display_times) > 15:
            self._recent_display_times.pop(0)
        if len(self._recent_display_times) >= 2:
            dt = self._recent_display_times[-1] - self._recent_display_times[0]
            if dt > 0.01:
                self.display_fps = round((len(self._recent_display_times) - 1) / dt, 1)

        # Determine LIVE vs DEGRADED based on frame age
        status = "LIVE" if age_ms < 350.0 else "DEGRADED"
        return self.jpeg_bytes, self.timestamp, round(age_ms, 1), status


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
        self.last_valid_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self.trajectory: List[Dict[str, float]] = []
        self.nav_path: List[Dict[str, float]] = []
        self.waypoints: List[Dict[str, float]] = []
        self.active_goal: Optional[Dict[str, float]] = None
        self.nav_stats: Dict[str, Any] = {
            "dist_to_goal_m": 0.0,
            "current_waypoint_idx": 0,
            "total_waypoints": 0,
            "replan_count": 0,
            "cmd_vx": 0.0,
            "cmd_wz": 0.0,
            "path_valid": False,
            "terrain_cost": 0.0,
            "slope_deg": 0.0,
            "clearance_m": 1.0,
            "speed_scale": 1.0,
            "nav_reason": "Nominal path following",
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

        # 6b. Vehicle Geometry & Passage Diagnostics
        self.vehicle_diagnostics: Dict[str, Any] = {
            "vehicle": {
                "length_m": 0.56,
                "width_m": 0.48,
                "height_m": 0.26,
                "inscribed_radius_m": 0.24,
                "circumscribed_radius_m": 0.369,
                "nominal_passage_m": 0.68,
                "tight_passage_limit_m": 0.58,
                "min_turn_diameter_m": 0.94,
                "safety_margin_m": 0.10,
            },
            "status": "SAFE",
            "can_fit": True,
            "can_turn": True,
            "available_clear_width_m": 1.20,
            "required_clear_width_m": 0.68,
            "clearance_margin_m": 0.72,
            "turning_diameter_available_m": 1.20,
        }

        # 6c. YOLO Perception Diagnostics
        self.yolo_diagnostics: Dict[str, Any] = {
            "model": "yolov8n.onnx (fallback-saliency)",
            "enabled": True,
            "device": "CPU",
            "fps": 0.0,
            "latency_ms": 0.0,
            "num_detections": 0,
            "detections": [],
            "status": "ONLINE",
            "backend": "cv2.dnn",
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

        # 8. Visual Stream Buffers (Latest-Frame Bounded Buffers, size=1)
        self.streams: Dict[str, StreamBuffer] = {
            "raw": StreamBuffer("raw", target_fps=15.0),
            "perception": StreamBuffer("perception", target_fps=10.0),
            "segmentation": StreamBuffer("segmentation", target_fps=10.0),
            "vo": StreamBuffer("vo", target_fps=10.0),
            "chase": StreamBuffer("chase", target_fps=15.0),
            "yolo": StreamBuffer("yolo", target_fps=10.0),
        }

        # 9. Rolling Event Log
        self.events: List[Dict[str, Any]] = []
        self.add_event("SYSTEM", "NAVIGUARD Operator Dashboard initialized.")

        # 10. Structured Failure & Success Records
        self.failure_record: Optional[Dict[str, Any]] = None
        self.success_record: Optional[Dict[str, Any]] = None

        # 11. Additional Map Layers Data
        self.checkpoints: List[Dict[str, Any]] = []
        self.persistent_obstacles: List[Dict[str, Any]] = []
        self.traversability_grid: Optional[List[int]] = None

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
            self.last_valid_pose = dict(self.robot_pose)
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

    def set_waypoints(self, wps: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.waypoints = wps

    def set_checkpoints(self, ckpts: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.checkpoints = ckpts

    def set_persistent_obstacles(self, obsts: List[Dict[str, Any]]) -> None:
        with self._lock:
            self.persistent_obstacles = obsts

    def set_active_goal(self, goal: Optional[Dict[str, float]]) -> None:
        with self._lock:
            self.active_goal = goal

    def set_failure_record(self, record: Dict[str, Any]) -> None:
        """Store structured mission failure explanation and emit concise event."""
        with self._lock:
            self.failure_record = record
            self.mission_state = "FAILED"
        reason = record.get("human_reason", record.get("failure_code", "UNKNOWN"))
        code = record.get("failure_code", "GENERIC_FAILURE")
        rec = f"{record.get('recovery_attempts', 0)}/{record.get('recovery_max_attempts', 3)}"
        self.add_event("FAILURE", f"MISSION FAILED [{code}]: {reason} | Recovery: {rec}")

    def set_success_record(self, record: Dict[str, Any]) -> None:
        """Store structured mission success report and emit event."""
        with self._lock:
            self.success_record = record
            self.mission_state = "COMPLETED"
        dist = record.get("travel_distance_m", 0.0)
        dur = record.get("duration_sec", 0.0)
        self.add_event("MISSION", f"GOAL REACHED! Distance={dist:.2f}m, Time={dur:.1f}s")

    def clear_failure_and_success(self) -> None:
        with self._lock:
            self.failure_record = None
            self.success_record = None

    def set_vehicle_diagnostics(self, diag: Dict[str, Any]) -> None:
        with self._lock:
            self.vehicle_diagnostics.update(diag)

    def set_yolo_diagnostics(self, diag: Dict[str, Any]) -> None:
        with self._lock:
            self.yolo_diagnostics.update(diag)

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

            # Stream metrics summary
            streams_summary = {}
            for k, buf in self.streams.items():
                now = time.time()
                age_ms = max(0.0, (now - buf.timestamp) * 1000.0) if buf.timestamp > 0.0 else 0.0
                streams_summary[k] = {
                    "source_fps": buf.source_fps,
                    "display_fps": buf.display_fps,
                    "frame_age_ms": round(age_ms, 1),
                    "status": "LIVE" if age_ms < 350.0 else "DEGRADED",
                    "encode_latency_ms": buf.encode_latency_ms,
                    "seq": buf.frame_seq,
                }

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
                "last_valid_pose": dict(self.last_valid_pose),
                "trajectory": list(self.trajectory),
                "nav_path": list(self.nav_path),
                "waypoints": list(self.waypoints),
                "checkpoints": list(self.checkpoints),
                "persistent_obstacles": list(self.persistent_obstacles),
                "active_goal": dict(self.active_goal) if self.active_goal else None,
                "nav_stats": dict(self.nav_stats),
                "replan_diagnostics": dict(self.replan_diagnostics),
                "vehicle_diagnostics": dict(self.vehicle_diagnostics),
                "yolo_diagnostics": dict(self.yolo_diagnostics),
                "map_meta": dict(self.map_meta),
                "subsystems": dict(self.subsystems),
                "events": list(self.events),
                "stream_metrics": streams_summary,
                "failure_record": dict(self.failure_record) if self.failure_record else None,
                "success_record": dict(self.success_record) if self.success_record else None,
                "timestamp": time.time(),
            }

    def set_jpeg_frame(
        self,
        frame_type: str,
        jpeg_bytes: bytes,
        timestamp: Optional[float] = None,
        encode_latency_ms: float = 0.0,
        source_fps: float = 0.0,
    ) -> None:
        """Store new frame in bounded buffer (replaces older frame, size=1)."""
        ts = timestamp if timestamp is not None else time.time()
        with self._lock:
            if frame_type in self.streams:
                self.streams[frame_type].update_frame(
                    jpeg_bytes=jpeg_bytes,
                    timestamp=ts,
                    encode_latency_ms=encode_latency_ms,
                    source_fps=source_fps,
                )

    def get_jpeg_frame(self, frame_type: str, with_metadata: bool = False):
        """Get latest JPEG frame. If with_metadata=True, returns (bytes, ts, age_ms, status). Otherwise returns bytes or None."""
        with self._lock:
            if frame_type in self.streams:
                bytes_out, ts, age_ms, status = self.streams[frame_type].consume_latest()
                if with_metadata:
                    return bytes_out, ts, age_ms, status
                return bytes_out
            if with_metadata:
                return None, 0.0, 0.0, "DEGRADED"
            return None

    def get_stream_frame(self, frame_type: str) -> Tuple[Optional[bytes], float, float, str]:
        """Get latest stream frame with metadata (jpeg_bytes, timestamp, age_ms, status)."""
        return self.get_jpeg_frame(frame_type, with_metadata=True)
