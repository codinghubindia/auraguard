"""Keyframe representation for NAVIGUARD Visual-Inertial Graph-SLAM.

Stores metric poses in map and odom frames, extracted ORB visual features,
descriptors, visual landmark associations, and depth estimates.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Keyframe:
    """Represents a discrete keyframe in the SLAM pose graph."""

    def __init__(
        self,
        keyframe_id: int,
        stamp_sec: float,
        pose_map: Tuple[float, float, float],
        pose_odom: Tuple[float, float, float],
        keypoints: np.ndarray,
        descriptors: Optional[np.ndarray] = None,
        landmark_ids: Optional[List[int]] = None,
    ) -> None:
        self.keyframe_id = keyframe_id
        self.stamp_sec = stamp_sec
        # (x_m, y_m, theta_rad) in map frame
        self.pose_map = np.array(pose_map, dtype=np.float64)
        # (x_m, y_m, theta_rad) in odom frame
        self.pose_odom = np.array(pose_odom, dtype=np.float64)
        # Keypoints: (N, 2) array of (u, v) pixel coordinates
        self.keypoints = np.array(keypoints, dtype=np.float32) if len(keypoints) > 0 else np.empty((0, 2), dtype=np.float32)
        # Descriptors: (N, 32) uint8 ORB descriptors
        self.descriptors = descriptors if descriptors is not None else np.empty((0, 32), dtype=np.uint8)
        # Associated global landmark IDs (-1 if unassociated)
        self.landmark_ids = list(landmark_ids) if landmark_ids is not None else [-1] * len(self.keypoints)

    @property
    def num_features(self) -> int:
        return len(self.keypoints)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize keyframe metadata and features for map persistence."""
        return {
            'keyframe_id': self.keyframe_id,
            'stamp_sec': self.stamp_sec,
            'pose_map': self.pose_map.tolist(),
            'pose_odom': self.pose_odom.tolist(),
            'keypoints': self.keypoints.tolist(),
            'descriptors': self.descriptors.tolist() if self.descriptors is not None else [],
            'landmark_ids': self.landmark_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Keyframe':
        """Deserialize keyframe from dictionary."""
        descriptors = np.array(data['descriptors'], dtype=np.uint8) if data.get('descriptors') else None
        return cls(
            keyframe_id=int(data['keyframe_id']),
            stamp_sec=float(data['stamp_sec']),
            pose_map=tuple(data['pose_map']),
            pose_odom=tuple(data['pose_odom']),
            keypoints=np.array(data['keypoints'], dtype=np.float32),
            descriptors=descriptors,
            landmark_ids=data.get('landmark_ids'),
        )
