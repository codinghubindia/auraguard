"""Extended Kalman Filter State Estimator for NAVIGUARD UGV.

Fuses wheel odometry (linear velocity, yaw rate), IMU (yaw rate),
and visual motion measurements (optical yaw rate, scale-observed linear velocity)
using an Extended Kalman Filter with Mahalanobis gating and Joseph-form covariance updates.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from naviguard_state_estimation.state_model import KinematicStateModel, RobotState


class EstimatorConfig:
    """Configurable process noise and Mahalanobis gating thresholds."""

    def __init__(
        self,
        q_pos: float = 0.01,
        q_theta: float = 0.005,
        q_vx: float = 0.08,
        q_wz: float = 0.05,
        mahalanobis_gate_1d: float = 9.0,   # ~3 sigma for 1-DOF
        mahalanobis_gate_2d: float = 13.8,  # ~3.7 sigma for 2-DOF
    ) -> None:
        self.q_pos = q_pos
        self.q_theta = q_theta
        self.q_vx = q_vx
        self.q_wz = q_wz
        self.mahalanobis_gate_1d = mahalanobis_gate_1d
        self.mahalanobis_gate_2d = mahalanobis_gate_2d


class NaviguardStateEstimator:
    """Multi-sensor EKF state estimator maintaining local robot state in odom frame."""

    def __init__(self, config: Optional[EstimatorConfig] = None) -> None:
        self.config = config or EstimatorConfig()
        self.model = KinematicStateModel(
            q_pos=self.config.q_pos,
            q_theta=self.config.q_theta,
            q_vx=self.config.q_vx,
            q_wz=self.config.q_wz,
        )
        self.state = RobotState()
        self.is_initialized = False

        # Measurement tracking metrics
        self.updates_count = {'wheel': 0, 'imu': 0, 'visual': 0}
        self.rejections_count = {'wheel': 0, 'imu': 0, 'visual': 0}
        self.last_rejection_reason = {'wheel': "", 'imu': "", 'visual': ""}

    def initialize_pose(self, x: float, y: float, theta: float, stamp_sec: float) -> None:
        """Initialize robot state with known starting pose."""
        self.state = RobotState(x=x, y=y, theta=theta, vx=0.0, wz=0.0, stamp_sec=stamp_sec)
        self.is_initialized = True

    def predict_to_time(self, target_time_sec: float) -> None:
        """Advance state prediction up to target_time_sec."""
        if not self.is_initialized:
            self.state.stamp_sec = target_time_sec
            self.is_initialized = True
            return

        dt = target_time_sec - self.state.stamp_sec
        if dt > 1e-5:
            self.state = self.model.predict(self.state, dt)

    def update_wheel_odometry(
        self,
        vx: float,
        wz: float,
        vx_cov: float,
        wz_cov: float,
        stamp_sec: float,
    ) -> Tuple[bool, str]:
        """Incorporate wheel metric linear and angular velocity measurement."""
        self.predict_to_time(stamp_sec)

        # Measurement vector: z = [vx, wz]^T
        z = np.array([vx, wz], dtype=np.float64)
        # Measurement matrix: H = [0, 0, 0, 1, 0; 0, 0, 0, 0, 1]
        H = np.array([
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        R = np.diag([max(vx_cov, 1e-4), max(wz_cov, 1e-4)])

        # Innovation: y = z - H * x
        y = z - H @ self.state.vec
        S = H @ self.state.cov @ H.T + R

        # Mahalanobis gating test
        try:
            S_inv = np.linalg.inv(S)
            d_mahalanobis = float(y.T @ S_inv @ y)
            if d_mahalanobis > self.config.mahalanobis_gate_2d:
                self.rejections_count['wheel'] += 1
                reason = f"MAHALANOBIS_GATE_EXCEEDED_{d_mahalanobis:.1f}"
                self.last_rejection_reason['wheel'] = reason
                return False, reason
        except np.linalg.LinAlgError:
            return False, "SINGULAR_INNOVATION_COVARIANCE"

        # Kalman update
        K = self.state.cov @ H.T @ S_inv
        self.state.vec += K @ y
        self.state.vec[2] = self.model._wrap_angle(self.state.vec[2])

        # Joseph form covariance update: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        I = np.eye(5)
        IKH = I - K @ H
        self.state.cov = IKH @ self.state.cov @ IKH.T + K @ R @ K.T

        self.updates_count['wheel'] += 1
        return True, "ACCEPTED"

    def update_imu(self, wz: float, wz_cov: float, stamp_sec: float) -> Tuple[bool, str]:
        """Incorporate IMU angular yaw rate measurement."""
        self.predict_to_time(stamp_sec)

        z = np.array([wz], dtype=np.float64)
        H = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
        R = np.array([[max(wz_cov, 1e-4)]], dtype=np.float64)

        y = z - H @ self.state.vec
        S = H @ self.state.cov @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
            d_mahalanobis = float(y.T @ S_inv @ y)
            if d_mahalanobis > self.config.mahalanobis_gate_1d:
                self.rejections_count['imu'] += 1
                reason = f"MAHALANOBIS_GATE_EXCEEDED_{d_mahalanobis:.1f}"
                self.last_rejection_reason['imu'] = reason
                return False, reason
        except np.linalg.LinAlgError:
            return False, "SINGULAR_INNOVATION_COVARIANCE"

        K = self.state.cov @ H.T @ S_inv
        self.state.vec += (K @ y).flatten()
        self.state.vec[2] = self.model._wrap_angle(self.state.vec[2])

        I = np.eye(5)
        IKH = I - K @ H
        self.state.cov = IKH @ self.state.cov @ IKH.T + K @ R @ K.T

        self.updates_count['imu'] += 1
        return True, "ACCEPTED"

    def update_visual(
        self,
        wz_vis: float,
        wz_cov: float,
        stamp_sec: float,
        metric_vx: Optional[float] = None,
        vx_cov: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Incorporate visual yaw rate and optional scale-observed linear velocity."""
        self.predict_to_time(stamp_sec)

        if metric_vx is not None and vx_cov is not None:
            # 2-DOF update: [vx, wz]
            z = np.array([metric_vx, wz_vis], dtype=np.float64)
            H = np.array([
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ], dtype=np.float64)
            R = np.diag([max(vx_cov, 1e-4), max(wz_cov, 1e-4)])
            gate_thresh = self.config.mahalanobis_gate_2d
        else:
            # 1-DOF update: only wz
            z = np.array([wz_vis], dtype=np.float64)
            H = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
            R = np.array([[max(wz_cov, 1e-4)]], dtype=np.float64)
            gate_thresh = self.config.mahalanobis_gate_1d

        y = z - H @ self.state.vec
        S = H @ self.state.cov @ H.T + R

        try:
            S_inv = np.linalg.inv(S)
            d_mahalanobis = float(y.T @ S_inv @ y)
            if d_mahalanobis > gate_thresh:
                self.rejections_count['visual'] += 1
                reason = f"MAHALANOBIS_GATE_EXCEEDED_{d_mahalanobis:.1f}"
                self.last_rejection_reason['visual'] = reason
                return False, reason
        except np.linalg.LinAlgError:
            return False, "SINGULAR_INNOVATION_COVARIANCE"

        K = self.state.cov @ H.T @ S_inv
        self.state.vec += (K @ y).flatten()
        self.state.vec[2] = self.model._wrap_angle(self.state.vec[2])

        I = np.eye(5)
        IKH = I - K @ H
        self.state.cov = IKH @ self.state.cov @ IKH.T + K @ R @ K.T

        self.updates_count['visual'] += 1
        return True, "ACCEPTED"
