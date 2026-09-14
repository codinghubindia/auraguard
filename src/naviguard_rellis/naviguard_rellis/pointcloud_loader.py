"""
PointCloud Loader for RELLIS-3D Dataset.

Lightweight parser for PCD files (ASCII and Binary) supporting XYZ and Intensity.
Creates standard sensor_msgs/msg/PointCloud2 messages for visualization and evaluation.
"""

import os
import struct
from typing import Dict, List, Optional, Tuple
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def parse_pcd_header(lines: List[str]) -> Tuple[Dict, int]:
    """Parse PCD header lines and return header dictionary and data line/byte offset."""
    header = {
        'version': '0.7',
        'fields': [],
        'size': [],
        'type': [],
        'count': [],
        'width': 0,
        'height': 1,
        'viewpoint': [0, 0, 0, 1, 0, 0, 0],
        'points': 0,
        'data': 'ascii'
    }
    header_lines_count = 0

    for idx, line in enumerate(lines):
        header_lines_count += 1
        line_str = line.strip()
        if not line_str or line_str.startswith('#'):
            continue
        parts = line_str.split()
        tag = parts[0].upper()

        if tag == 'VERSION':
            header['version'] = parts[1]
        elif tag == 'FIELDS':
            header['fields'] = parts[1:]
        elif tag == 'SIZE':
            header['size'] = [int(x) for x in parts[1:]]
        elif tag == 'TYPE':
            header['type'] = parts[1:]
        elif tag == 'COUNT':
            header['count'] = [int(x) for x in parts[1:]]
        elif tag == 'WIDTH':
            header['width'] = int(parts[1])
        elif tag == 'HEIGHT':
            header['height'] = int(parts[1])
        elif tag == 'POINTS':
            header['points'] = int(parts[1])
        elif tag == 'DATA':
            header['data'] = parts[1].lower()
            break

    return header, header_lines_count


class PointCloudLoader:
    """Loads point clouds from PCD files."""

    @staticmethod
    def load_pcd(filepath: str) -> Optional[np.ndarray]:
        """Load XYZ(I) points from a PCD file into an Nx3 or Nx4 numpy float32 array."""
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'rb') as f:
                header_lines = []
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line_str = line.decode('utf-8', errors='ignore')
                    header_lines.append(line_str)
                    if line_str.strip().startswith('DATA'):
                        break

                header, _ = parse_pcd_header(header_lines)
                num_points = header.get('points', 0)
                if num_points <= 0:
                    num_points = header.get('width', 0) * header.get('height', 1)

                data_type = header.get('data', 'ascii')

                if data_type == 'ascii':
                    points = []
                    for line in f:
                        parts = line.decode('utf-8', errors='ignore').strip().split()
                        if len(parts) >= 3:
                            try:
                                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            except ValueError:
                                continue
                    return np.array(points, dtype=np.float32)

                elif data_type == 'binary':
                    raw_data = f.read()
                    point_step = sum(header['size']) if header['size'] else 16
                    dtype_list = []
                    for f_name, size, t_type in zip(header['fields'], header['size'], header['type']):
                        np_type = 'f4' if t_type in ('F', 'f') else 'u1'
                        dtype_list.append((f_name, np_type))

                    arr = np.frombuffer(raw_data[:num_points * point_step], dtype=dtype_list)
                    if 'x' in arr.dtype.names and 'y' in arr.dtype.names and 'z' in arr.dtype.names:
                        xyz = np.column_stack([arr['x'], arr['y'], arr['z']]).astype(np.float32)
                        return xyz
                    return None
        except Exception:
            return None

    @staticmethod
    def points_to_pointcloud2_msg(
        points: np.ndarray,
        frame_id: str = 'base_link',
        stamp=None
    ) -> PointCloud2:
        """Convert an Nx3 float32 numpy array to a sensor_msgs/msg/PointCloud2."""
        msg = PointCloud2()
        if stamp is not None:
            msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = len(points)
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = points.astype(np.float32).tobytes()
        return msg
