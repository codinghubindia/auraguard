"""State representation and kinematic motion model for NAVIGUARD UGV.

Defines the 5-DOF local state vector [x, y, theta, vx, wz]^T in the odom reference frame,
state covariance matrices, and non-linear unicycle kinematic propagation.
"""

from typing import Optional, Tuple
import numpy as np


class RobotState:
    """Represents the 5-DOF kinematic state and covariance in odom frame."""

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        vx: float = 0.0,
        wz: float = 0.0,
        stamp_sec: float = 0.0,
    ) -> None:
        self.stamp_sec = stamp_sec
        # State vector: [x (m), y (m), theta (rad), vx (m/s), wz (rad/s)]
        self.vec = np.array([x, y, theta, vx, wz], dtype=np.float64)
        # Covariance matrix 5x5
        self.cov = np.diag([0.01, 0.01, 0.005, 0.04, 0.02]).astype(np.float64)

    @property
    def x(self) -> float:
        return float(self.vec[0])

    @property
    def y(self) -> float:
        return float(self.vec[1])

    @property
    def theta(self) -> float:
        return float(self.vec[2])

    @property
    def vx(self) -> float:
        return float(self.vec[3])

    @property
    def wz(self) -> float:
        return float(self.vec[4])


class KinematicStateModel:
    """Non-linear unicycle process model and Jacobian propagation."""

    def __init__(
        self,
        q_pos: float = 0.01,
        q_theta: float = 0.005,
        q_vx: float = 0.10,
        q_wz: float = 0.08,
    ) -> None:
        self.Q_base = np.diag([q_pos, q_pos, q_theta, q_vx, q_wz]).astype(np.float64)

    def predict(self, state: RobotState, dt: float) -> RobotState:
        """Propagate state forward by dt using kinematic unicycle motion model."""
        if dt <= 0.0:
            return state

        x, y, theta, vx, wz = state.vec

        # Midpoint yaw angle during the step
        theta_mid = theta + wz * (dt / 2.0)
        c_mid = np.cos(theta_mid)
        s_mid = np.sin(theta_mid)

        # Non-linear state update
        x_new = x + vx * c_mid * dt
        y_new = y + vx * s_mid * dt
        theta_new = self._wrap_angle(theta + wz * dt)
        vx_new = vx
        wz_new = wz

        # State transition Jacobian F_x = df/dx
        F_x = np.eye(5, dtype=np.float64)
        F_x[0, 2] = -vx * s_mid * dt
        F_x[0, 3] = c_mid * dt
        F_x[0, 4] = -vx * s_mid * (dt * dt / 2.0)

        F_x[1, 2] = vx * c_mid * dt
        F_x[1, 3] = s_mid * dt
        F_x[1, 4] = vx * c_mid * (dt * dt / 2.0)

        F_x[2, 4] = dt

        # Propagate covariance: P = F * P * F^T + Q * dt
        Q = self.Q_base * dt
        cov_new = F_x @ state.cov @ F_x.T + Q

        new_state = RobotState(x_new, y_new, theta_new, vx_new, wz_new, state.stamp_sec + dt)
        new_state.cov = cov_new
        return new_state

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float(np.arctan2(np.sin(angle), np.cos(angle)))
