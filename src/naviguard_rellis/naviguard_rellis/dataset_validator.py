"""
RELLIS-3D Dataset Validator.

Validates sequence structure, file availability, frame counts, and formatting.
Provides a clear diagnostic report without failing silently.
"""

import os
import sys
import glob
from typing import Dict, List, Optional


IMAGE_SUBDIRS = ['pylon_camera_node', 'cam0', 'images', 'image', 'pylon_camera_node_label_id']


def find_image_directory(seq_path: str) -> Optional[str]:
    """Find the directory containing RGB images in the sequence path."""
    for subdir in IMAGE_SUBDIRS:
        cand = os.path.join(seq_path, subdir)
        if os.path.isdir(cand):
            # Check if it has jpg or png files
            images = glob.glob(os.path.join(cand, '*.jpg')) + glob.glob(os.path.join(cand, '*.png'))
            if images:
                return cand
    return None


class DatasetValidator:
    """Validates sequence integrity and data files."""

    def __init__(self, data_root: str, sequence: str = "00000"):
        self.data_root = os.path.abspath(os.path.expanduser(data_root))
        self.sequence = sequence
        self.seq_path = os.path.join(self.data_root, self.sequence)

    def validate(self) -> Dict:
        """
        Validate the dataset sequence.
        Returns a dictionary with validation results.
        """
        issues: List[str] = []
        info = {
            'valid': False,
            'data_root': self.data_root,
            'sequence': self.sequence,
            'seq_path': self.seq_path,
            'has_seq_dir': False,
            'image_dir': None,
            'image_count': 0,
            'has_calibration': False,
            'has_poses': False,
            'pose_count': 0,
            'has_pointcloud': False,
            'pcd_count': 0,
            'issues': issues
        }

        if not os.path.exists(self.data_root):
            issues.append(f"Data root directory does not exist: {self.data_root}")
            return info

        if not os.path.exists(self.seq_path):
            issues.append(f"Sequence directory does not exist: {self.seq_path}")
            return info

        info['has_seq_dir'] = True

        # Check images
        img_dir = find_image_directory(self.seq_path)
        if img_dir:
            info['image_dir'] = img_dir
            images = glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png'))
            info['image_count'] = len(images)
            if len(images) == 0:
                issues.append(f"Image directory found at {img_dir}, but no image files found.")
        else:
            issues.append(f"No image directory found in {self.seq_path}. Expected one of {IMAGE_SUBDIRS}")

        # Check calibration
        calib_file = os.path.join(self.seq_path, 'camera_info.txt')
        if os.path.isfile(calib_file):
            info['has_calibration'] = True
        else:
            # Check parent directory
            calib_parent = os.path.join(self.data_root, 'camera_info.txt')
            if os.path.isfile(calib_parent):
                info['has_calibration'] = True
            else:
                issues.append("camera_info.txt not found in sequence or root directory.")

        # Check poses
        poses_file = os.path.join(self.seq_path, 'poses.txt')
        if os.path.isfile(poses_file):
            info['has_poses'] = True
            try:
                with open(poses_file, 'r') as f:
                    poses = [line.strip() for line in f if line.strip()]
                info['pose_count'] = len(poses)
            except Exception as e:
                issues.append(f"Error reading poses.txt: {e}")
        else:
            issues.append("poses.txt not found in sequence directory.")

        # Check PCDs
        pcd_dir = os.path.join(self.seq_path, 'pcd')
        if os.path.isdir(pcd_dir):
            pcds = glob.glob(os.path.join(pcd_dir, '*.pcd'))
            info['has_pointcloud'] = len(pcds) > 0
            info['pcd_count'] = len(pcds)

        # Overall validity: Valid if seq_dir exists, images > 0, and poses > 0
        info['valid'] = (
            info['has_seq_dir'] and
            info['image_count'] > 0 and
            info['has_poses'] and
            info['pose_count'] > 0
        )

        return info

    def print_report(self) -> None:
        """Print validation diagnostic report."""
        report = self.validate()
        print("=" * 60)
        print("RELLIS-3D DATASET VALIDATION REPORT")
        print("=" * 60)
        print(f"Data Root:       {report['data_root']}")
        print(f"Sequence:        {report['sequence']}")
        print(f"Sequence Exists: {'YES' if report['has_seq_dir'] else 'NO'}")
        print(f"Images Found:    {report['image_count']} (Dir: {report['image_dir']})")
        print(f"Calibration:     {'FOUND' if report['has_calibration'] else 'NOT FOUND'}")
        print(f"Poses:           {'FOUND' if report['has_poses'] else 'NOT FOUND'} ({report['pose_count']} poses)")
        print(f"Point Clouds:    {'FOUND' if report['has_pointcloud'] else 'NOT FOUND'} ({report['pcd_count']} PCDs)")
        print(f"Overall Valid:   {'YES' if report['valid'] else 'NO'}")
        if report['issues']:
            print("-" * 60)
            print("Issues / Notes:")
            for issue in report['issues']:
                print(f"  - {issue}")
        print("=" * 60)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "~/datasets/rellis3d"
    seq = sys.argv[2] if len(sys.argv) > 2 else "00000"
    validator = DatasetValidator(root, seq)
    validator.print_report()


if __name__ == '__main__':
    main()
