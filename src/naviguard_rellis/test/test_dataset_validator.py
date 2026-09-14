import os
import tempfile
from naviguard_rellis.dataset_validator import DatasetValidator


def test_dataset_validator_nonexistent():
    validator = DatasetValidator("/non/existent/path", "00000")
    report = validator.validate()
    assert report['valid'] is False
    assert len(report['issues']) > 0


def test_dataset_validator_synthetic_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        seq_dir = os.path.join(tmpdir, "00000")
        img_dir = os.path.join(seq_dir, "pylon_camera_node")
        os.makedirs(img_dir)

        # Create dummy image files
        for i in range(5):
            with open(os.path.join(img_dir, f"frame_{i:04d}.jpg"), 'w') as f:
                f.write("dummy")

        # Create dummy poses
        with open(os.path.join(seq_dir, "poses.txt"), 'w') as f:
            for _ in range(5):
                f.write("1 0 0 0 0 1 0 0 0 0 1 0\n")

        # Create dummy calib
        with open(os.path.join(seq_dir, "camera_info.txt"), 'w') as f:
            f.write("width: 640\nheight: 480\n")

        validator = DatasetValidator(tmpdir, "00000")
        report = validator.validate()
        assert report['valid'] is True
        assert report['image_count'] == 5
        assert report['pose_count'] == 5
        assert report['has_calibration'] is True
