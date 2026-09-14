import numpy as np
import pytest

from naviguard_slam.keyframe import Keyframe
from naviguard_slam.place_recognition import PlaceRecognizer


def test_place_recognition_gap_and_distance_gating():
    recognizer = PlaceRecognizer(min_keyframe_gap=6, search_radius_m=3.0, min_inliers=15)

    kpts = np.random.uniform(50, 400, (20, 2)).astype(np.float32)
    descs = np.random.randint(0, 255, (20, 32), dtype=np.uint8)

    kf0 = Keyframe(0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), kpts, descs)
    # kf2 has gap of only 2 (< 6)
    kf2 = Keyframe(2, 2.0, (0.1, 0.1, 0.0), (0.1, 0.1, 0.0), kpts, descs)
    # kf8 has gap of 8 but is 10 meters away (> 3.0 m)
    kf8 = Keyframe(8, 8.0, (10.0, 10.0, 0.0), (10.0, 10.0, 0.0), kpts, descs)

    cand_close = recognizer.detect_loop_closure(kf2, [kf0])
    assert cand_close is None  # rejected due to keyframe gap < 6

    cand_far = recognizer.detect_loop_closure(kf8, [kf0])
    assert cand_far is None  # rejected due to spatial distance > 3.0 m


def test_place_recognition_matching():
    recognizer = PlaceRecognizer(min_keyframe_gap=5, search_radius_m=3.0, min_inliers=10)

    # Identical features at close spatial proximity
    kpts = np.random.uniform(50, 400, (30, 2)).astype(np.float32)
    descs = np.random.randint(0, 255, (30, 32), dtype=np.uint8)

    kf0 = Keyframe(0, 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), kpts, descs)
    kf6 = Keyframe(6, 6.0, (0.5, 0.2, 0.1), (0.5, 0.2, 0.1), kpts, descs)

    loop_cand = recognizer.detect_loop_closure(kf6, [kf0])
    assert loop_cand is not None
    assert loop_cand.to_kf_id == 0
    assert loop_cand.inlier_count >= 10
