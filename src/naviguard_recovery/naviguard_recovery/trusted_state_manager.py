"""Trusted State and Checkpoint Manager for Autonomous Recovery.

Buffers trustworthy historical navigation states during nominal operation and
applies multi-criteria scoring to select the optimal recovery checkpoint without
using simulator ground truth.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrustedCheckpoint:
    """Historical navigation state captured during high-confidence operation."""
    checkpoint_id: int
    timestamp_sec: float
    map_pose: Tuple[float, float, float]    # (x, y, yaw) in map frame
    odom_pose: Tuple[float, float, float]   # (x, y, yaw) in odom frame
    confidence_overall: float
    confidence_loc: float
    confidence_vis: float
    velocity: Tuple[float, float]           # (vx, wz)
    map_clearance_m: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp_sec": float(self.timestamp_sec),
            "map_pose": [float(v) for v in self.map_pose],
            "odom_pose": [float(v) for v in self.odom_pose],
            "confidence_overall": float(self.confidence_overall),
            "confidence_loc": float(self.confidence_loc),
            "confidence_vis": float(self.confidence_vis),
            "velocity": [float(v) for v in self.velocity],
            "map_clearance_m": float(self.map_clearance_m),
        }


@dataclass
class TrustedStateManagerConfig:
    max_checkpoints: int = 40
    min_displacement_m: float = 0.25
    min_rotation_rad: float = 0.25
    min_confidence_overall: float = 0.82
    min_confidence_loc: float = 0.75
    min_confidence_vis: float = 0.70
    optimal_backtrack_dist_m: float = 0.80
    max_backtrack_dist_m: float = 3.00


class TrustedStateManager:
    """Manages the lifecycle of trusted navigation checkpoints."""

    def __init__(self, config: Optional[TrustedStateManagerConfig] = None) -> None:
        self.cfg = config or TrustedStateManagerConfig()
        self.checkpoints: deque[TrustedCheckpoint] = deque(maxlen=self.cfg.max_checkpoints)
        self.next_checkpoint_id: int = 0
        self.selected_checkpoint: Optional[TrustedCheckpoint] = None

    def maybe_add_checkpoint(
        self,
        timestamp_sec: float,
        map_pose: Tuple[float, float, float],
        odom_pose: Tuple[float, float, float],
        conf_overall: float,
        conf_loc: float,
        conf_vis: float,
        velocity: Tuple[float, float],
        phase7_state: str,
        map_clearance_m: float = 1.0,
    ) -> bool:
        """Evaluate whether current state qualifies as a trustworthy checkpoint."""
        if phase7_state.upper() != "CONTINUE":
            return False

        if (
            conf_overall < self.cfg.min_confidence_overall
            or conf_loc < self.cfg.min_confidence_loc
            or conf_vis < self.cfg.min_confidence_vis
        ):
            return False

        if map_clearance_m < 0.40:
            return False

        # Displacement gating relative to latest stored checkpoint
        if len(self.checkpoints) > 0:
            last = self.checkpoints[-1]
            dx = map_pose[0] - last.map_pose[0]
            dy = map_pose[1] - last.map_pose[1]
            dist = float(np.hypot(dx, dy))
            dtheta = abs(float(np.arctan2(np.sin(map_pose[2] - last.map_pose[2]), np.cos(map_pose[2] - last.map_pose[2]))))

            if dist < self.cfg.min_displacement_m and dtheta < self.cfg.min_rotation_rad:
                return False

        ckpt = TrustedCheckpoint(
            checkpoint_id=self.next_checkpoint_id,
            timestamp_sec=timestamp_sec,
            map_pose=map_pose,
            odom_pose=odom_pose,
            confidence_overall=conf_overall,
            confidence_loc=conf_loc,
            confidence_vis=conf_vis,
            velocity=velocity,
            map_clearance_m=map_clearance_m,
        )
        self.next_checkpoint_id += 1
        self.checkpoints.append(ckpt)
        return True

    def select_best_checkpoint(
        self,
        current_pose: Tuple[float, float, float],
        current_time: float,
        grid_data: Optional[List[int]] = None,
        grid_res: float = 0.05,
        grid_w: int = 0,
        grid_h: int = 0,
        grid_ox: float = 0.0,
        grid_oy: float = 0.0,
    ) -> Optional[TrustedCheckpoint]:
        """Score and return the optimal checkpoint for recovery."""
        if not self.checkpoints:
            self.selected_checkpoint = None
            return None

        best_score = -1.0
        best_ckpt: Optional[TrustedCheckpoint] = None

        for ckpt in self.checkpoints:
            dx = current_pose[0] - ckpt.map_pose[0]
            dy = current_pose[1] - ckpt.map_pose[1]
            dist = float(np.hypot(dx, dy))

            # Exclude checkpoints outside allowable backtrack envelope
            if dist > self.cfg.max_backtrack_dist_m:
                continue

            # Check obstacle clearance in grid map if map is provided
            if grid_data and grid_w > 0 and grid_h > 0:
                cx = int((ckpt.map_pose[0] - grid_ox) / max(0.01, grid_res))
                cy = int((ckpt.map_pose[1] - grid_oy) / max(0.01, grid_res))
                if 0 <= cx < grid_w and 0 <= cy < grid_h:
                    occ = grid_data[cy * grid_w + cx]
                    if occ >= 50:  # Obstacle cell
                        continue

            # Gaussian distance optimality centered at optimal_backtrack_dist_m
            f_dist = float(np.exp(-((dist - self.cfg.optimal_backtrack_dist_m) ** 2) / (2.0 * (0.6 ** 2))))

            # Recency scoring
            dt = max(0.0, current_time - ckpt.timestamp_sec)
            f_time = float(np.exp(-dt / 20.0))

            # Composite score
            score = (
                0.35 * ckpt.confidence_loc
                + 0.25 * ckpt.confidence_overall
                + 0.25 * f_dist
                + 0.15 * f_time
            )

            if score > best_score:
                best_score = score
                best_ckpt = ckpt

        # If all candidates exceeded max distance, fall back to newest valid
        if best_ckpt is None and self.checkpoints:
            best_ckpt = self.checkpoints[-1]

        self.selected_checkpoint = best_ckpt
        return best_ckpt
