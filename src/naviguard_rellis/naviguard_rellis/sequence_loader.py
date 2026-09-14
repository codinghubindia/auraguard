"""
Sequence Loader for RELLIS-3D Dataset.

Coordinates image loading, timestamps, calibration, and ground-truth pose synchronization.
"""

import os
import glob
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np

from naviguard_rellis.calibration_loader import parse_rellis_camera_info, create_camera_info_msg
from naviguard_rellis.pose_loader import PoseLoader
from naviguard_rellis.dataset_validator import find_image_directory


@dataclass
class SequenceFrame:
    frame_idx: int
    image_path: str
    timestamp_sec: float
    gt_pose: Optional[Dict]
    camera_info_params: Dict


class SequenceLoader:
    """Loads all frames from a RELLIS-3D sequence."""

    def __init__(self, data_root: str, sequence: str = "00000", fps: float = 10.0):
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        self.sequence = sequence
        self.fps = fps
        self.seq_path = os.path.join(self.data_root, self.sequence)

        self.frames: List[SequenceFrame] = []
        self.is_loaded = False
        self.calib_params: Dict = {}
        self.pose_loader: Optional[PoseLoader] = None

        self._load()

    def _load(self) -> None:
        if not os.path.isdir(self.seq_path):
            return

        # Load calibration
        calib_file = os.path.join(self.seq_path, 'camera_info.txt')
        if not os.path.isfile(calib_file):
            calib_file = os.path.join(self.data_root, 'camera_info.txt')
        self.calib_params = parse_rellis_camera_info(calib_file)

        # Load poses
        poses_file = os.path.join(self.seq_path, 'poses.txt')
        self.pose_loader = PoseLoader(poses_file)

        # Load timestamps if present
        timestamps = []
        ts_file = os.path.join(self.seq_path, 'camera_timestamp.txt')
        if not os.path.isfile(ts_file):
            ts_file = os.path.join(self.seq_path, 'timestamps.txt')
        if os.path.isfile(ts_file):
            try:
                with open(ts_file, 'r') as f:
                    for line in f:
                        l = line.strip()
                        if l:
                            timestamps.append(float(l))
            except Exception:
                timestamps = []

        # Find image files
        img_dir = find_image_directory(self.seq_path)
        if not img_dir:
            return

        img_files = sorted(
            glob.glob(os.path.join(img_dir, '*.jpg')) +
            glob.glob(os.path.join(img_dir, '*.png'))
        )

        dt = 1.0 / max(1.0, self.fps)
        for idx, img_path in enumerate(img_files):
            if idx < len(timestamps):
                t_sec = timestamps[idx]
            else:
                t_sec = idx * dt

            gt_pose = self.pose_loader.get_pose(idx) if (self.pose_loader and idx < len(self.pose_loader)) else None

            self.frames.append(SequenceFrame(
                frame_idx=idx,
                image_path=img_path,
                timestamp_sec=t_sec,
                gt_pose=gt_pose,
                camera_info_params=self.calib_params
            ))

        self.is_loaded = len(self.frames) > 0

    def __len__(self) -> int:
        return len(self.frames)

    def get_frame(self, index: int) -> Optional[SequenceFrame]:
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None
