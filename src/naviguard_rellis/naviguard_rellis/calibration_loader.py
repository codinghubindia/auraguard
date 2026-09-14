"""
Calibration Loader for RELLIS-3D Dataset.

Parses camera calibration files into standard ROS 2 sensor_msgs/msg/CameraInfo messages.
Handles both RELLIS-3D camera_info.txt format and YAML/custom formats with safe defaults.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from sensor_msgs.msg import CameraInfo


def parse_k_matrix_from_text(text: str) -> Optional[np.ndarray]:
    """Parse 3x3 intrinsic matrix K from text lines."""
    numbers = []
    for token in text.replace('[', ' ').replace(']', ' ').replace(',', ' ').split():
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    if len(numbers) >= 9:
        return np.array(numbers[:9], dtype=np.float64).reshape((3, 3))
    return None


def parse_rellis_camera_info(filepath: str) -> Dict:
    """
    Parse a RELLIS-3D camera_info.txt file.

    Expected typical contents:
    width: 1920 (or 1280)
    height: 1200 (or 720)
    camera_matrix: ...
    distortion_coefficients: ...
    projection_matrix: ...
    """
    params = {
        'width': 1920,
        'height': 1200,
        'k': np.array([
            [1400.0, 0.0, 960.0],
            [0.0, 1400.0, 600.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64),
        'd': [0.0, 0.0, 0.0, 0.0, 0.0],
        'distortion_model': 'plumb_bob',
        'r': np.eye(3, dtype=np.float64),
        'p': np.array([
            [1400.0, 0.0, 960.0, 0.0],
            [0.0, 1400.0, 600.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float64),
    }

    if not os.path.exists(filepath):
        return params

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Check for plain whitespace-separated numbers (e.g. official RELLIS camera_info.txt: fx fy cx cy)
        all_tokens = []
        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith('#'):
                continue
            for t in line_str.split():
                try:
                    all_tokens.append(float(t))
                except ValueError:
                    pass

        if len(all_tokens) == 4 and ':' not in "".join(lines):
            fx, fy, cx, cy = all_tokens[0], all_tokens[1], all_tokens[2], all_tokens[3]
            params['k'] = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            params['p'] = np.array([
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ], dtype=np.float64)
            return params

        current_key = None
        buffer_numbers = []

        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith('#'):
                continue

            if ':' in line_str:
                key, val = line_str.split(':', 1)
                key = key.strip().lower()
                val = val.strip()

                if key in ('width', 'image_width'):
                    try:
                        params['width'] = int(val)
                    except ValueError:
                        pass
                elif key in ('height', 'image_height'):
                    try:
                        params['height'] = int(val)
                    except ValueError:
                        pass
                elif key in ('distortion_model', 'model'):
                    params['distortion_model'] = val if val else 'plumb_bob'
                elif 'k' in key or 'camera_matrix' in key or 'intrinsic' in key:
                    current_key = 'k'
                    buffer_numbers = []
                    for t in val.replace('[', ' ').replace(']', ' ').replace(',', ' ').split():
                        try:
                            buffer_numbers.append(float(t))
                        except ValueError:
                            pass
                elif 'd' in key or 'distortion' in key:
                    current_key = 'd'
                    buffer_numbers = []
                    for t in val.replace('[', ' ').replace(']', ' ').replace(',', ' ').split():
                        try:
                            buffer_numbers.append(float(t))
                        except ValueError:
                            pass
                elif 'p' in key or 'projection' in key:
                    current_key = 'p'
                    buffer_numbers = []
                    for t in val.replace('[', ' ').replace(']', ' ').replace(',', ' ').split():
                        try:
                            buffer_numbers.append(float(t))
                        except ValueError:
                            pass
            else:
                if current_key:
                    for t in line_str.replace('[', ' ').replace(']', ' ').replace(',', ' ').split():
                        try:
                            buffer_numbers.append(float(t))
                        except ValueError:
                            pass

            if current_key == 'k' and len(buffer_numbers) >= 9:
                params['k'] = np.array(buffer_numbers[:9], dtype=np.float64).reshape((3, 3))
                current_key = None
            elif current_key == 'd' and len(buffer_numbers) >= 5:
                params['d'] = buffer_numbers
                current_key = None
            elif current_key == 'p' and len(buffer_numbers) >= 12:
                params['p'] = np.array(buffer_numbers[:12], dtype=np.float64).reshape((3, 4))
                current_key = None

    except Exception:
        pass

    return params


def create_camera_info_msg(
    params: Dict,
    frame_id: str = 'camera_optical_link',
    timestamp=None
) -> CameraInfo:
    """Construct a sensor_msgs/msg/CameraInfo from parsed parameters."""
    msg = CameraInfo()
    if timestamp is not None:
        msg.header.stamp = timestamp
    msg.header.frame_id = frame_id
    msg.width = int(params.get('width', 1920))
    msg.height = int(params.get('height', 1200))
    msg.distortion_model = params.get('distortion_model', 'plumb_bob')

    d = params.get('d', [0.0, 0.0, 0.0, 0.0, 0.0])
    msg.d = [float(x) for x in d]

    k = params.get('k')
    if isinstance(k, np.ndarray):
        msg.k = k.flatten().tolist()
    else:
        msg.k = [1400.0, 0.0, 960.0, 0.0, 1400.0, 600.0, 0.0, 0.0, 1.0]

    r = params.get('r')
    if isinstance(r, np.ndarray):
        msg.r = r.flatten().tolist()
    else:
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    p = params.get('p')
    if isinstance(p, np.ndarray):
        msg.p = p.flatten().tolist()
    else:
        msg.p = [
            msg.k[0], msg.k[1], msg.k[2], 0.0,
            msg.k[3], msg.k[4], msg.k[5], 0.0,
            msg.k[6], msg.k[7], msg.k[8], 0.0
        ]

    return msg
