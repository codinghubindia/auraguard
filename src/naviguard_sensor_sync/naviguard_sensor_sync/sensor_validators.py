"""Sensor Data Format, Intrinsics, and Field Validators for NAVIGUARD UGV.

Validates CameraInfo calibration matrix, IMU orientation/acceleration fields,
and Odometry frames and kinematic structures.
"""

from typing import List, Optional, Tuple
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, Imu


class CameraCalibrationReport:
    """Report detailing camera intrinsic parameters and calibration validity."""

    def __init__(
        self,
        is_valid: bool = False,
        width: int = 0,
        height: int = 0,
        fx: float = 0.0,
        fy: float = 0.0,
        cx: float = 0.0,
        cy: float = 0.0,
        distortion_model: str = "",
        distortion_coeffs: Optional[List[float]] = None,
        frame_id: str = "",
        dimension_match: bool = True,
        status: str = "NO_DATA",
        message: str = "",
    ) -> None:
        self.is_valid = is_valid
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion_model = distortion_model
        self.distortion_coeffs = distortion_coeffs or []
        self.frame_id = frame_id
        self.dimension_match = dimension_match
        self.status = status
        self.message = message


class ImuValidationReport:
    """Report detailing IMU fields, covariance, and orientation availability."""

    def __init__(
        self,
        is_valid: bool = False,
        frame_id: str = "",
        has_angular_velocity: bool = False,
        angular_vel_norm: float = 0.0,
        has_linear_acceleration: bool = False,
        linear_accel_norm: float = 0.0,
        orientation_status: str = "UNKNOWN",
        orientation_quat: Optional[Tuple[float, float, float, float]] = None,
        has_valid_covariance: bool = False,
        status: str = "NO_DATA",
        message: str = "",
    ) -> None:
        self.is_valid = is_valid
        self.frame_id = frame_id
        self.has_angular_velocity = has_angular_velocity
        self.angular_vel_norm = angular_vel_norm
        self.has_linear_acceleration = has_linear_acceleration
        self.linear_accel_norm = linear_accel_norm
        self.orientation_status = orientation_status
        self.orientation_quat = orientation_quat
        self.has_valid_covariance = has_valid_covariance
        self.status = status
        self.message = message


class OdometryValidationReport:
    """Report detailing Odometry header frames, child frame, and twist/pose integrity."""

    def __init__(
        self,
        is_valid: bool = False,
        header_frame_id: str = "",
        child_frame_id: str = "",
        has_valid_pose: bool = False,
        has_valid_twist: bool = False,
        linear_speed: float = 0.0,
        angular_speed: float = 0.0,
        status: str = "NO_DATA",
        message: str = "",
    ) -> None:
        self.is_valid = is_valid
        self.header_frame_id = header_frame_id
        self.child_frame_id = child_frame_id
        self.has_valid_pose = has_valid_pose
        self.has_valid_twist = has_valid_twist
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.status = status
        self.message = message


def validate_camera_info(
    info_msg: CameraInfo,
    image_msg: Optional[Image] = None,
) -> CameraCalibrationReport:
    """Validate camera_info intrinsics matrix, principal point, and image dimensions."""
    if len(info_msg.k) < 9:
        return CameraCalibrationReport(
            is_valid=False,
            status="FAIL",
            message="Camera matrix K has fewer than 9 elements.",
        )

    fx = float(info_msg.k[0])
    cx = float(info_msg.k[2])
    fy = float(info_msg.k[4])
    cy = float(info_msg.k[5])
    w = int(info_msg.width)
    h = int(info_msg.height)

    # Intrinsic validity checks
    is_fx_valid = fx > 10.0 and not np.isnan(fx)
    is_fy_valid = fy > 10.0 and not np.isnan(fy)
    is_cx_valid = (0.0 < cx < w) and not np.isnan(cx) if w > 0 else False
    is_cy_valid = (0.0 < cy < h) and not np.isnan(cy) if h > 0 else False
    has_pos_dim = (w > 0) and (h > 0)

    dim_match = True
    if image_msg is not None:
        dim_match = (image_msg.width == w) and (image_msg.height == h)

    is_valid = is_fx_valid and is_fy_valid and is_cx_valid and is_cy_valid and has_pos_dim and dim_match

    status = "OK" if is_valid else "FAIL"
    reasons = []
    if not is_fx_valid or not is_fy_valid:
        reasons.append("Invalid focal length")
    if not is_cx_valid or not is_cy_valid:
        reasons.append("Principal point outside frame")
    if not has_pos_dim:
        reasons.append("Zero/negative resolution")
    if not dim_match:
        reasons.append("Image resolution mismatch with camera_info")

    msg_str = "Valid camera calibration" if is_valid else ", ".join(reasons)

    return CameraCalibrationReport(
        is_valid=is_valid,
        width=w,
        height=h,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        distortion_model=info_msg.distortion_model,
        distortion_coeffs=list(info_msg.d),
        frame_id=info_msg.header.frame_id,
        dimension_match=dim_match,
        status=status,
        message=msg_str,
    )


def validate_imu(msg: Imu) -> ImuValidationReport:
    """Validate IMU angular velocities, linear accelerations, and orientation status."""
    # Check angular velocity
    av = msg.angular_velocity
    av_vals = [av.x, av.y, av.z]
    has_av = all(not np.isnan(v) and not np.isinf(v) for v in av_vals)
    av_norm = float(np.linalg.norm(av_vals)) if has_av else 0.0

    # Check linear acceleration
    la = msg.linear_acceleration
    la_vals = [la.x, la.y, la.z]
    has_la = all(not np.isnan(v) and not np.isinf(v) for v in la_vals)
    la_norm = float(np.linalg.norm(la_vals)) if has_la else 0.0

    # Check orientation field explicitly
    q = msg.orientation
    q_vals = (q.x, q.y, q.z, q.w)
    q_norm = np.linalg.norm(q_vals)

    # In standard ROS, orientation_covariance[0] == -1 indicates orientation is NOT provided
    if msg.orientation_covariance[0] == -1.0 or q_norm < 1e-4:
        orientation_status = "UNAVAILABLE"
        quat = None
    elif abs(q_norm - 1.0) < 1e-2:
        orientation_status = "AVAILABLE"
        quat = q_vals
    else:
        orientation_status = "INVALID"
        quat = q_vals

    # Check covariances (standard non-negative diagonals if provided)
    has_valid_cov = (
        msg.angular_velocity_covariance[0] >= 0.0
        and msg.linear_acceleration_covariance[0] >= 0.0
    )

    is_valid = has_av and has_la and (orientation_status != "INVALID")
    status = "OK" if is_valid else "FAIL"
    msg_str = f"IMU fields verified (orientation: {orientation_status})"

    return ImuValidationReport(
        is_valid=is_valid,
        frame_id=msg.header.frame_id,
        has_angular_velocity=has_av,
        angular_vel_norm=av_norm,
        has_linear_acceleration=has_la,
        linear_accel_norm=la_norm,
        orientation_status=orientation_status,
        orientation_quat=quat,
        has_valid_covariance=has_valid_cov,
        status=status,
        message=msg_str,
    )


def validate_odometry(msg: Odometry) -> OdometryValidationReport:
    """Validate Odometry frames, position/orientation, and velocity twists."""
    header_frame = msg.header.frame_id
    child_frame = msg.child_frame_id

    # Check pose
    p = msg.pose.pose.position
    p_vals = [p.x, p.y, p.z]
    q = msg.pose.pose.orientation
    q_vals = [q.x, q.y, q.z, q.w]
    has_valid_pose = (
        all(not np.isnan(v) and not np.isinf(v) for v in p_vals)
        and all(not np.isnan(v) and not np.isinf(v) for v in q_vals)
        and (abs(np.linalg.norm(q_vals) - 1.0) < 1e-2 or np.linalg.norm(q_vals) < 1e-4)
    )

    # Check twist
    v = msg.twist.twist.linear
    w = msg.twist.twist.angular
    lin_speed = float(np.hypot(v.x, v.y))
    ang_speed = float(abs(w.z))
    has_valid_twist = (
        all(not np.isnan(val) and not np.isinf(val) for val in [v.x, v.y, v.z, w.x, w.y, w.z])
    )

    # Expected standard frames: header='odom', child='base_link'
    expected_frames = (header_frame == "odom") and (child_frame == "base_link")

    is_valid = has_valid_pose and has_valid_twist and expected_frames
    status = "OK" if is_valid else "FAIL"

    reasons = []
    if not expected_frames:
        reasons.append(f"Frames '{header_frame}' -> '{child_frame}' unexpected")
    if not has_valid_pose:
        reasons.append("Invalid pose")
    if not has_valid_twist:
        reasons.append("Invalid twist")

    msg_str = "Valid odometry message" if is_valid else ", ".join(reasons)

    return OdometryValidationReport(
        is_valid=is_valid,
        header_frame_id=header_frame,
        child_frame_id=child_frame,
        has_valid_pose=has_valid_pose,
        has_valid_twist=has_valid_twist,
        linear_speed=lin_speed,
        angular_speed=ang_speed,
        status=status,
        message=msg_str,
    )
