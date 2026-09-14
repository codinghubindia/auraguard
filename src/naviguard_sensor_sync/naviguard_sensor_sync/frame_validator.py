"""TF Frame and Extrinsic Calibration Validator for NAVIGUARD UGV.

Inspects TF tree hierarchy, verifies sensor frames, validates camera optical frame
convention (REP-103), and compiles machine-readable extrinsic calibration reports.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import rclpy
from rclpy.time import Time
import tf2_ros
from geometry_msgs.msg import TransformStamped


class FrameValidationReport:
    """Report detailing availability of TF frames, transforms, and optical conventions."""

    def __init__(
        self,
        all_frames_available: bool = False,
        missing_frames: Optional[List[str]] = None,
        camera_optical_convention_valid: bool = False,
        extrinsics: Optional[Dict[str, Any]] = None,
        status: str = "NO_DATA",
        message: str = "",
    ) -> None:
        self.all_frames_available = all_frames_available
        self.missing_frames = missing_frames or []
        self.camera_optical_convention_valid = camera_optical_convention_valid
        self.extrinsics = extrinsics or {}
        self.status = status
        self.message = message


class FrameValidator:
    """Validates TF coordinate frame tree and static sensor extrinsics."""

    EXPECTED_FRAMES = [
        "odom",
        "base_link",
        "base_footprint",
        "camera_link",
        "camera_optical_link",
        "imu_link",
        "left_front_wheel",
        "right_front_wheel",
        "left_rear_wheel",
        "right_rear_wheel",
    ]

    def __init__(self, tf_buffer: tf2_ros.Buffer) -> None:
        self.tf_buffer = tf_buffer

    def validate_frames(self, lookup_time: Optional[Time] = None) -> FrameValidationReport:
        """Query TF buffer to verify required transforms and validate optical conventions."""
        time_point = lookup_time if lookup_time is not None else Time()

        missing: List[str] = []
        extrinsics: Dict[str, Any] = {}

        # 1. Test presence of base_link to each required link
        for frame in self.EXPECTED_FRAMES:
            if frame == "base_link":
                continue

            target = "base_link"
            source = frame
            # For odom, relationship is odom -> base_link
            if frame == "odom":
                target = "odom"
                source = "base_link"

            try:
                tf_msg = self.tf_buffer.lookup_transform(
                    target,
                    source,
                    time_point,
                    timeout=rclpy.duration.Duration(seconds=0.05),
                )
                trans = tf_msg.transform.translation
                rot = tf_msg.transform.rotation
                extrinsics[f"{source}_to_{target}"] = {
                    "translation": [trans.x, trans.y, trans.z],
                    "rotation_xyzw": [rot.x, rot.y, rot.z, rot.w],
                }
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                missing.append(frame)

        # 2. Specifically validate camera_link -> camera_optical_link convention (REP-103)
        optical_valid = False
        try:
            opt_tf = self.tf_buffer.lookup_transform(
                "camera_link",
                "camera_optical_link",
                time_point,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
            q = opt_tf.transform.rotation
            R = self._quat_to_rot_matrix(q.x, q.y, q.z, q.w)

            # In standard ROS optical convention:
            # optical +Z (forward) maps to camera_link +X (forward): R @ [0,0,1]^T ~ [1,0,0]^T
            # optical +X (right) maps to camera_link -Y (right):     R @ [1,0,0]^T ~ [0,-1,0]^T
            # optical +Y (down) maps to camera_link -Z (down):       R @ [0,1,0]^T ~ [0,0,-1]^T
            z_optical_in_cam = R @ np.array([0, 0, 1], dtype=np.float64)
            x_optical_in_cam = R @ np.array([1, 0, 0], dtype=np.float64)
            y_optical_in_cam = R @ np.array([0, 1, 0], dtype=np.float64)

            is_z_fwd = np.allclose(z_optical_in_cam, [1, 0, 0], atol=0.05)
            is_x_right = np.allclose(x_optical_in_cam, [0, -1, 0], atol=0.05)
            is_y_down = np.allclose(y_optical_in_cam, [0, 0, -1], atol=0.05)

            optical_valid = is_z_fwd and is_x_right and is_y_down
            extrinsics["camera_optical_convention"] = {
                "valid_rep103": optical_valid,
                "z_optical_direction_in_cam_link": z_optical_in_cam.tolist(),
                "x_optical_direction_in_cam_link": x_optical_in_cam.tolist(),
                "y_optical_direction_in_cam_link": y_optical_in_cam.tolist(),
            }
        except Exception as e:
            extrinsics["camera_optical_convention"] = {
                "valid_rep103": False,
                "error": str(e),
            }

        all_avail = len(missing) == 0
        if all_avail and optical_valid:
            status = "OK"
            msg = "All expected frames present and camera optical frame REP-103 verified"
        elif all_avail and not optical_valid:
            status = "WARN"
            msg = "All frames present but camera optical frame deviates from REP-103"
        else:
            status = "WARN" if len(missing) <= 2 else "FAIL"
            msg = f"Missing transforms for: {', '.join(missing)}"

        return FrameValidationReport(
            all_frames_available=all_avail,
            missing_frames=missing,
            camera_optical_convention_valid=optical_valid,
            extrinsics=extrinsics,
            status=status,
            message=msg,
        )

    @staticmethod
    def _quat_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
        norm = np.hypot(np.hypot(x, y), np.hypot(z, w))
        if norm > 0.0:
            x, y, z, w = x / norm, y / norm, z / norm, w / norm

        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ], dtype=np.float64)
