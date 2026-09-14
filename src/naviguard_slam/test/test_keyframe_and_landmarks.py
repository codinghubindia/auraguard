import pytest
import numpy as np
import cv2

from naviguard_slam.keyframe import Keyframe
from naviguard_slam.landmark_database import VisualLandmark, LandmarkDatabase


def test_keyframe_creation_and_serialization():
    kpts = np.array([[100.0, 150.0], [200.0, 250.0]], dtype=np.float32)
    descs = np.random.randint(0, 255, (2, 32), dtype=np.uint8)

    kf = Keyframe(
        keyframe_id=1,
        stamp_sec=10.5,
        pose_map=(1.0, 2.0, 0.5),
        pose_odom=(0.8, 1.9, 0.4),
        keypoints=kpts,
        descriptors=descs,
        landmark_ids=[10, 11],
    )

    assert kf.keyframe_id == 1
    assert kf.num_features == 2
    assert kf.pose_map[0] == 1.0

    d = kf.to_dict()
    assert d['keyframe_id'] == 1
    assert d['stamp_sec'] == 10.5

    kf_restored = Keyframe.from_dict(d)
    assert kf_restored.keyframe_id == 1
    assert np.allclose(kf_restored.pose_map, kf.pose_map)
    assert np.allclose(kf_restored.keypoints, kf.keypoints)
    assert np.array_equal(kf_restored.descriptors, kf.descriptors)


def test_landmark_creation_and_observations():
    desc = np.random.randint(0, 255, 32, dtype=np.uint8)
    lm = VisualLandmark(landmark_id=42, position_map=(3.0, 4.0, 1.2), descriptor=desc, first_keyframe_id=0)

    assert lm.landmark_id == 42
    assert lm.observation_count == 1
    assert lm.observing_keyframes == [0]

    lm.add_observation(keyframe_id=1)
    assert lm.observation_count == 2
    assert lm.observing_keyframes == [0, 1]

    d = lm.to_dict()
    lm_restored = VisualLandmark.from_dict(d)
    assert lm_restored.landmark_id == 42
    assert lm_restored.observation_count == 2
    assert np.allclose(lm_restored.position, lm.position)


def test_landmark_database_matching():
    db = LandmarkDatabase()
    desc1 = np.zeros(32, dtype=np.uint8)
    desc2 = np.full(32, 255, dtype=np.uint8)

    id1 = db.add_landmark((1.0, 1.0, 0.5), desc1, keyframe_id=0)
    id2 = db.add_landmark((2.0, 2.0, 0.5), desc2, keyframe_id=0)

    assert len(db) == 2

    # Query with identical descriptor to desc1
    query_descs = np.array([desc1], dtype=np.uint8)
    matches = db.match_query_frame(query_descs)
    assert len(matches) == 1
    assert matches[0][1] == id1  # matched landmark id1
