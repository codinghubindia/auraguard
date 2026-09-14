"""Visual Place Recognition and Loop Closure Candidate Detector for NAVIGUARD SLAM.

Detects trajectory revisits using spatial-index gating, ORB visual feature matching,
and 2D/3D geometric RANSAC verification to find loop-closure constraints.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from naviguard_slam.keyframe import Keyframe


class LoopClosureConstraint:
    """Represents a geometrically verified loop-closure edge between two keyframes."""

    def __init__(
        self,
        from_kf_id: int,
        to_kf_id: int,
        relative_pose: Tuple[float, float, float],  # (dx, dy, dtheta)
        inlier_count: int,
        confidence_score: float,
    ) -> None:
        self.from_kf_id = from_kf_id
        self.to_kf_id = to_kf_id
        # (dx, dy, dtheta) in from_kf reference frame
        self.relative_pose = np.array(relative_pose, dtype=np.float64)
        self.inlier_count = inlier_count
        self.confidence_score = confidence_score


class PlaceRecognizer:
    """Evaluates loop closure candidates and computes rigid relative transforms."""

    def __init__(
        self,
        min_keyframe_gap: int = 8,
        search_radius_m: float = 3.5,
        min_inliers: int = 15,
        match_ratio_thresh: float = 0.75,
    ) -> None:
        self.min_keyframe_gap = min_keyframe_gap
        self.search_radius_m = search_radius_m
        self.min_inliers = min_inliers
        self.match_ratio_thresh = match_ratio_thresh
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_loop_closure(
        self,
        current_kf: Keyframe,
        past_keyframes: List[Keyframe],
    ) -> Optional[LoopClosureConstraint]:
        """Search past keyframes for visual loop-closure with current keyframe."""
        if current_kf.descriptors is None or len(current_kf.descriptors) < self.min_inliers:
            return None

        cur_x, cur_y = current_kf.pose_map[0], current_kf.pose_map[1]

        best_candidate: Optional[LoopClosureConstraint] = None
        max_inliers: int = 0

        for past_kf in past_keyframes:
            # 1. Keyframe gap condition (exclude immediate temporal neighbors)
            if abs(current_kf.keyframe_id - past_kf.keyframe_id) < self.min_keyframe_gap:
                continue

            # 2. Spatial proximity gate
            past_x, past_y = past_kf.pose_map[0], past_kf.pose_map[1]
            dist = float(np.hypot(cur_x - past_x, cur_y - past_y))
            if dist > self.search_radius_m:
                continue

            if past_kf.descriptors is None or len(past_kf.descriptors) < self.min_inliers:
                continue

            # 3. Feature matching with Lowe's ratio test
            matches = self.matcher.knnMatch(current_kf.descriptors, past_kf.descriptors, k=2)
            good_matches = []
            for m_tuple in matches:
                if len(m_tuple) == 2:
                    m, n = m_tuple
                    if m.distance < self.match_ratio_thresh * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self.min_inliers:
                continue

            # 4. Geometric verification using 2D Affine RANSAC
            pts_cur = np.float32([current_kf.keypoints[m.queryIdx] for m in good_matches])
            pts_past = np.float32([past_kf.keypoints[m.trainIdx] for m in good_matches])

            transform_matrix, inliers = cv2.estimateAffinePartial2D(
                pts_cur, pts_past, method=cv2.RANSAC, ransacReprojThreshold=5.0
            )

            if transform_matrix is None or inliers is None:
                continue

            num_inliers = int(np.sum(inliers))
            if num_inliers >= self.min_inliers and num_inliers > max_inliers:
                max_inliers = num_inliers
                # Extract relative yaw from affine matrix
                s_cos = transform_matrix[0, 0]
                s_sin = transform_matrix[1, 0]
                dtheta = float(np.arctan2(s_sin, s_cos))

                # Expected spatial relative offset in current_kf frame
                th_cur = current_kf.pose_map[2]
                c, s = np.cos(th_cur), np.sin(th_cur)
                glob_dx = past_x - cur_x
                glob_dy = past_y - cur_y
                rel_dx = c * glob_dx + s * glob_dy
                rel_dy = -s * glob_dx + c * glob_dy

                conf = float(num_inliers) / max(len(good_matches), 1)
                best_candidate = LoopClosureConstraint(
                    from_kf_id=current_kf.keyframe_id,
                    to_kf_id=past_kf.keyframe_id,
                    relative_pose=(rel_dx, rel_dy, dtheta),
                    inlier_count=num_inliers,
                    confidence_score=conf,
                )

        return best_candidate
