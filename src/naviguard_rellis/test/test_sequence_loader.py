import os
import tempfile
from naviguard_rellis.sequence_loader import SequenceLoader


def test_sequence_loader_with_fixture():
    with tempfile.TemporaryDirectory() as tmpdir:
        seq_dir = os.path.join(tmpdir, "00000")
        img_dir = os.path.join(seq_dir, "pylon_camera_node")
        os.makedirs(img_dir)

        # Create dummy image files
        for i in range(3):
            with open(os.path.join(img_dir, f"frame_{i:04d}.jpg"), 'w') as f:
                f.write("dummy")

        with open(os.path.join(seq_dir, "poses.txt"), 'w') as f:
            for i in range(3):
                f.write(f"1 0 0 {i} 0 1 0 0 0 0 1 0\n")

        with open(os.path.join(seq_dir, "camera_info.txt"), 'w') as f:
            f.write("width: 640\nheight: 480\n")

        loader = SequenceLoader(tmpdir, "00000", fps=10.0)
        assert loader.is_loaded is True
        assert len(loader) == 3
        f0 = loader.get_frame(0)
        assert f0 is not None
        assert f0.frame_idx == 0
        assert f0.gt_pose is not None
        assert f0.gt_pose['position'][0] == 0.0
