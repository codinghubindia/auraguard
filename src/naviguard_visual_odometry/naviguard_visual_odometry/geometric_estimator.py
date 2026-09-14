"""Geometric Visual Motion Estimator for NAVIGUARD UGV.

Computes 2-view relative camera rotation (R) and unit translation direction (t)
using the Essential Matrix (E) with RANSAC outlier rejection and cheirality verification.

IMPORTANT MONOCULAR SCALE NOTICE:
In a monocular camera setup without metric depth or calibrated external scale,
the recovered translation vector 't' represents ONLY a unit-scale direction (||t|| = 1).
It is NEVER assumed to be in meters, and translation_scale_status is strictly UNKNOWN.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class GeometricEstimatorConfig:
    """Configuration parameters for Essential matrix estimation and pose recovery."""

    def __init__(
        self,
        ransac_threshold: float = 1.0,
        ransac_prob: float = 0.999,
        min_inliers: int = 15,
        min_baseline_disp_px: float = 0.6,
    ) -> None:
        self.ransac_threshold = ransac_threshold
        self.ransac_prob = ransac_prob
        self.min_inliers = min_inliers
        self.min_baseline_disp_px = min_baseline_disp_px


class GeometricMotionEstimate:
    """Data container for 2-view relative camera motion estimation results."""

    def __init__(
        self,
        is_valid: bool = False,
        status: str = "INITIALIZING",
        R: Optional[np.ndarray] = None,
        unit_t: Optional[np.ndarray] = None,
        num_inliers: int = 0,
        inlier_ratio: float = 0.0,
        rot_angle_deg: float = 0.0,
        rot_axis: Optional[np.ndarray] = None,
        euler_angles_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        inlier_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.is_valid = is_valid
        self.status = status
        self.R = R if R is not None else np.eye(3, dtype=np.float64)
        self.unit_t = unit_t if unit_t is not None else np.zeros((3, 1), dtype=np.float64)
        self.num_inliers = num_inliers
        self.inlier_ratio = inlier_ratio
        self.rot_angle_deg = rot_angle_deg
        self.rot_axis = rot_axis if rot_axis is not None else np.zeros(3, dtype=np.float64)
        # Euler angles in degrees (pitch_x, yaw_y, roll_z in optical camera frame)
        self.pitch_deg, self.yaw_deg, self.roll_deg = euler_angles_deg
        self.inlier_mask = inlier_mask
        # Explicit monocular scale flag
        self.translation_scale_status = "UNKNOWN"

    @property
    def unit_tx(self) -> float:
        return float(self.unit_t[0, 0]) if self.unit_t.shape == (3, 1) else float(self.unit_t[0])

    @property
    def unit_ty(self) -> float:
        return float(self.unit_t[1, 0]) if self.unit_t.shape == (3, 1) else float(self.unit_t[1])

    @property
    def unit_tz(self) -> float:
        return float(self.unit_t[2, 0]) if self.unit_t.shape == (3, 1) else float(self.unit_t[2])


class GeometricMotionEstimator:
    """Estimates relative camera motion (R, unit-t) from 2D point correspondences."""

    def __init__(self, config: Optional[GeometricEstimatorConfig] = None) -> None:
        self.config = config or GeometricEstimatorConfig()

    def estimate(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        K: np.ndarray,
        median_displacement_px: float = 0.0,
    ) -> GeometricMotionEstimate:
        """Compute relative rotation and unit translation direction between two frames.

        Args:
            pts_prev: (N, 2) or (N, 1, 2) float32 coordinates in previous frame.
            pts_curr: (N, 2) or (N, 1, 2) float32 coordinates in current frame.
            K: (3, 3) float64 camera intrinsic calibration matrix.
            median_displacement_px: Median optical flow displacement across correspondences.

        Returns:
            GeometricMotionEstimate instance.
        """
        num_pts = len(pts_prev)
        if num_pts < self.config.min_inliers:
            return GeometricMotionEstimate(
                is_valid=False,
                status="INSUFFICIENT_CORRESPONDENCES",
                num_inliers=num_pts,
                inlier_ratio=0.0,
            )

        # Check for degenerate zero-baseline condition
        # If camera has not moved or motion is sub-pixel, epipolar geometry is degenerate (E = [t]x R -> 0)
        if median_displacement_px < self.config.min_baseline_disp_px:
            return GeometricMotionEstimate(
                is_valid=False,
                status="DEGENERATE_SMALL_BASELINE",
                num_inliers=num_pts,
                inlier_ratio=1.0,
            )

        pts0 = pts_prev.reshape(-1, 2).astype(np.float64)
        pts1 = pts_curr.reshape(-1, 2).astype(np.float64)

        # 1. Compute Essential Matrix with RANSAC
        E, essential_mask = cv2.findEssentialMat(
            pts0,
            pts1,
            cameraMatrix=K,
            method=cv2.RANSAC,
            prob=self.config.ransac_prob,
            threshold=self.config.ransac_threshold,
        )

        if E is None or E.shape != (3, 3) or essential_mask is None:
            return GeometricMotionEstimate(
                is_valid=False,
                status="ESSENTIAL_MATRIX_FAILED",
                num_inliers=0,
                inlier_ratio=0.0,
            )

        # 2. Recover Relative Pose (R, unit-t) with Cheirality Verification
        num_inliers, R, t, pose_mask = cv2.recoverPose(
            E,
            pts0,
            pts1,
            cameraMatrix=K,
            mask=essential_mask,
        )

        inlier_ratio = float(num_inliers) / float(max(num_pts, 1))

        if num_inliers < self.config.min_inliers or inlier_ratio < 0.25:
            return GeometricMotionEstimate(
                is_valid=False,
                status="LOW_GEOMETRIC_INLIERS",
                num_inliers=num_inliers,
                inlier_ratio=inlier_ratio,
                inlier_mask=pose_mask,
            )

        # Normalize translation vector explicitly to unit norm
        norm_t = np.linalg.norm(t)
        if norm_t > 1e-8:
            unit_t = t / norm_t
        else:
            unit_t = np.zeros((3, 1), dtype=np.float64)

        # 3. Calculate Rotation Representations
        # Axis-Angle representation via Rodrigues formula
        rot_vec, _ = cv2.Rodrigues(R)
        rot_angle_rad = float(np.linalg.norm(rot_vec))
        rot_angle_deg = float(np.degrees(rot_angle_rad))
        if rot_angle_rad > 1e-6:
            rot_axis = (rot_vec / rot_angle_rad).flatten()
        else:
            rot_axis = np.zeros(3, dtype=np.float64)

        # Euler angles in optical camera frame:
        # X: right, Y: down, Z: forward
        # Pitch: rotation about X, Yaw: rotation about Y, Roll: rotation about Z
        euler_deg = self._rotation_matrix_to_euler_deg(R)

        return GeometricMotionEstimate(
            is_valid=True,
            status="GEOMETRIC_MOTION_OK",
            R=R,
            unit_t=unit_t,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            rot_angle_deg=rot_angle_deg,
            rot_axis=rot_axis,
            euler_angles_deg=euler_deg,
            inlier_mask=pose_mask,
        )

    @staticmethod
    def _rotation_matrix_to_euler_deg(R: np.ndarray) -> Tuple[float, float, float]:
        """Decompose rotation matrix R into (pitch_x, yaw_y, roll_z) in degrees.

        Camera Optical Frame:
        +X: Right -> Pitch
        +Y: Down  -> Yaw (positive turns camera right toward -X)
        +Z: Forward -> Roll
        """
        sy = np.hypot(R[0, 0], R[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0.0

        return (
            float(np.degrees(pitch)),
            float(np.degrees(yaw)),
            float(np.degrees(roll)),
        )
