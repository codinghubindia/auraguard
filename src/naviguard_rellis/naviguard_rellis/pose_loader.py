"""
Pose Loader for RELLIS-3D Dataset.

Parses KITTI-format poses.txt (12 numbers per line representing 3x4 [R|t] transformation matrices)
into SE(3) positions and orientations (quaternions) for evaluation comparison.
"""

import os
import math
from typing import List, Optional, Tuple, Dict
import numpy as np
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header


def rot_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert a 3x3 rotation matrix to quaternion (qx, qy, qz, qw).
    Uses Shepperd's algorithm for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm > 1e-9:
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm
    else:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

    return qx, qy, qz, qw


def parse_kitti_pose_line(line: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Parse a single 12-element line into (R 3x3, t 3x1)."""
    parts = line.strip().split()
    if len(parts) < 12:
        return None
    try:
        vals = [float(p) for p in parts[:12]]
    except ValueError:
        return None

    mat = np.array(vals, dtype=np.float64).reshape((3, 4))
    R = mat[:, :3]
    t = mat[:, 3]
    return R, t


class PoseLoader:
    """Loads and formats poses from poses.txt."""

    def __init__(self, poses_file: str):
        self.poses_file = poses_file
        self.poses_se3: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.poses_file):
            return

        with open(self.poses_file, 'r') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            res = parse_kitti_pose_line(line)
            if res is None:
                continue
            R, t = res
            qx, qy, qz, qw = rot_matrix_to_quaternion(R)
            self.poses_se3.append({
                'index': idx,
                'position': (float(t[0]), float(t[1]), float(t[2])),
                'orientation': (qx, qy, qz, qw),
                'matrix': np.vstack([np.hstack([R, t.reshape(3, 1)]), [0, 0, 0, 1]]),
            })

    def __len__(self) -> int:
        return len(self.poses_se3)

    def get_pose(self, index: int) -> Optional[Dict]:
        if 0 <= index < len(self.poses_se3):
            return self.poses_se3[index]
        return None

    def to_pose_msg(self, index: int) -> Optional[Pose]:
        entry = self.get_pose(index)
        if entry is None:
            return None
        p = Pose()
        p.position.x = entry['position'][0]
        p.position.y = entry['position'][1]
        p.position.z = entry['position'][2]
        p.orientation.x = entry['orientation'][0]
        p.orientation.y = entry['orientation'][1]
        p.orientation.z = entry['orientation'][2]
        p.orientation.w = entry['orientation'][3]
        return p

    def to_path_msg(self, frame_id: str = 'odom', max_poses: Optional[int] = None) -> Path:
        path = Path()
        path.header.frame_id = frame_id
        count = len(self.poses_se3) if max_poses is None else min(len(self.poses_se3), max_poses)
        for i in range(count):
            entry = self.poses_se3[i]
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.pose.position.x = entry['position'][0]
            ps.pose.position.y = entry['position'][1]
            ps.pose.position.z = entry['position'][2]
            ps.pose.orientation.x = entry['orientation'][0]
            ps.pose.orientation.y = entry['orientation'][1]
            ps.pose.orientation.z = entry['orientation'][2]
            ps.pose.orientation.w = entry['orientation'][3]
            path.poses.append(ps)
        return path
