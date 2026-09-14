"""Pose Graph Optimizer (PGO) for NAVIGUARD Visual-Inertial SLAM.

Solves non-linear least squares optimization over 2D SE(2) pose graph edges
(sequential odometry constraints and loop-closure constraints) using SciPy with robust Huber loss.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import least_squares

from naviguard_slam.keyframe import Keyframe
from naviguard_slam.place_recognition import LoopClosureConstraint


def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


class GraphEdge:
    """Directed edge in SE(2) pose graph."""

    def __init__(
        self,
        from_idx: int,
        to_idx: int,
        dx: float,
        dy: float,
        dtheta: float,
        is_loop_closure: bool = False,
        weight_trans: float = 1.0,
        weight_rot: float = 2.0,
    ) -> None:
        self.from_idx = from_idx
        self.to_idx = to_idx
        self.dx = dx
        self.dy = dy
        self.dtheta = wrap_angle(dtheta)
        self.is_loop_closure = is_loop_closure
        self.weight_trans = weight_trans
        self.weight_rot = weight_rot


class PoseGraphOptimizer:
    """Maintains and optimizes keyframe pose graph."""

    def __init__(self) -> None:
        self.edges: List[GraphEdge] = []
        self.loop_closures_count: int = 0
        self.last_opt_error: float = 0.0

    def add_odometry_edge(
        self,
        from_idx: int,
        to_idx: int,
        dx: float,
        dy: float,
        dtheta: float,
    ) -> None:
        """Add sequential odometry constraint between keyframes."""
        edge = GraphEdge(
            from_idx=from_idx,
            to_idx=to_idx,
            dx=dx,
            dy=dy,
            dtheta=dtheta,
            is_loop_closure=False,
            weight_trans=1.0,
            weight_rot=2.0,
        )
        self.edges.append(edge)

    def add_loop_closure_edge(self, constraint: LoopClosureConstraint) -> None:
        """Add visual loop closure edge."""
        edge = GraphEdge(
            from_idx=constraint.from_kf_id,
            to_idx=constraint.to_kf_id,
            dx=float(constraint.relative_pose[0]),
            dy=float(constraint.relative_pose[1]),
            dtheta=float(constraint.relative_pose[2]),
            is_loop_closure=True,
            weight_trans=1.5 * constraint.confidence_score,
            weight_rot=3.0 * constraint.confidence_score,
        )
        self.edges.append(edge)
        self.loop_closures_count += 1

    def optimize(
        self,
        keyframes: List[Keyframe],
        max_nfev: int = 50,
    ) -> Tuple[bool, float]:
        """Optimize keyframe poses to satisfy odometry and loop-closure constraints."""
        n_keyframes = len(keyframes)
        if n_keyframes < 2 or not self.edges:
            return True, 0.0

        # Parameter vector: flatten [x_0, y_0, theta_0, x_1, y_1, theta_1, ...]
        x0 = np.zeros(3 * n_keyframes, dtype=np.float64)
        for i, kf in enumerate(keyframes):
            x0[3 * i] = kf.pose_map[0]
            x0[3 * i + 1] = kf.pose_map[1]
            x0[3 * i + 2] = kf.pose_map[2]

        anchor_pose = (float(x0[0]), float(x0[1]), float(x0[2]))

        def residuals_fun(params: np.ndarray) -> np.ndarray:
            res = []
            # 1. Anchor constraint on first keyframe (keep map origin fixed)
            res.append(100.0 * (params[0] - anchor_pose[0]))
            res.append(100.0 * (params[1] - anchor_pose[1]))
            res.append(100.0 * wrap_angle(params[2] - anchor_pose[2]))

            # 2. Graph edges
            for edge in self.edges:
                if edge.from_idx >= n_keyframes or edge.to_idx >= n_keyframes:
                    continue

                # Pose i
                xi = params[3 * edge.from_idx]
                yi = params[3 * edge.from_idx + 1]
                thi = params[3 * edge.from_idx + 2]

                # Pose j
                xj = params[3 * edge.to_idx]
                yj = params[3 * edge.to_idx + 1]
                thj = params[3 * edge.to_idx + 2]

                c, s = np.cos(thi), np.sin(thi)
                delta_x = xj - xi
                delta_y = yj - yi

                # Relative translation in frame i
                pred_dx = c * delta_x + s * delta_y
                pred_dy = -s * delta_x + c * delta_y
                pred_dth = wrap_angle(thj - thi)

                res.append(edge.weight_trans * (pred_dx - edge.dx))
                res.append(edge.weight_trans * (pred_dy - edge.dy))
                res.append(edge.weight_rot * wrap_angle(pred_dth - edge.dtheta))

            return np.array(res, dtype=np.float64)

        try:
            sol = least_squares(
                residuals_fun,
                x0,
                method='trf',
                loss='huber',
                f_scale=0.5,
                max_nfev=max_nfev,
            )

            # Update keyframe poses
            opt_params = sol.x
            for i, kf in enumerate(keyframes):
                kf.pose_map[0] = opt_params[3 * i]
                kf.pose_map[1] = opt_params[3 * i + 1]
                kf.pose_map[2] = wrap_angle(opt_params[3 * i + 2])

            self.last_opt_error = float(sol.cost)
            return True, self.last_opt_error
        except Exception:
            return False, 999.0
