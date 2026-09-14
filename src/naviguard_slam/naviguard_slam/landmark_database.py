"""Landmark database and feature triangulation for NAVIGUARD Visual-Inertial SLAM.

Maintains 3D visual landmarks in the map frame, tracks multi-view observations,
matches query visual features for localization and place recognition, and supports map serialization.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class VisualLandmark:
    """Represents a triangulated 3D visual landmark in the global map frame."""

    def __init__(
        self,
        landmark_id: int,
        position_map: Tuple[float, float, float],
        descriptor: np.ndarray,
        first_keyframe_id: int,
    ) -> None:
        self.landmark_id = landmark_id
        # [X_map, Y_map, Z_map]
        self.position = np.array(position_map, dtype=np.float64)
        # Representative 32-byte binary ORB descriptor
        self.descriptor = np.array(descriptor, dtype=np.uint8)
        self.observing_keyframes: List[int] = [first_keyframe_id]
        self.observation_count: int = 1

    def add_observation(self, keyframe_id: int, descriptor: Optional[np.ndarray] = None) -> None:
        """Add an observation from a new keyframe."""
        if keyframe_id not in self.observing_keyframes:
            self.observing_keyframes.append(keyframe_id)
            self.observation_count += 1
            if descriptor is not None:
                # Update descriptor with latest observation
                self.descriptor = np.array(descriptor, dtype=np.uint8)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'landmark_id': self.landmark_id,
            'position': self.position.tolist(),
            'descriptor': self.descriptor.tolist(),
            'observing_keyframes': self.observing_keyframes,
            'observation_count': self.observation_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualLandmark':
        lm = cls(
            landmark_id=int(data['landmark_id']),
            position_map=tuple(data['position']),
            descriptor=np.array(data['descriptor'], dtype=np.uint8),
            first_keyframe_id=int(data['observing_keyframes'][0]) if data['observing_keyframes'] else 0,
        )
        lm.observing_keyframes = list(data.get('observing_keyframes', []))
        lm.observation_count = int(data.get('observation_count', len(lm.observing_keyframes)))
        return lm


class LandmarkDatabase:
    """Repository storing and querying 3D landmarks in the map frame."""

    def __init__(self) -> None:
        self.landmarks: Dict[int, VisualLandmark] = {}
        self._next_id: int = 0
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def add_landmark(
        self,
        position_map: Tuple[float, float, float],
        descriptor: np.ndarray,
        keyframe_id: int,
    ) -> int:
        """Insert a newly triangulated 3D landmark."""
        lm_id = self._next_id
        self._next_id += 1
        lm = VisualLandmark(lm_id, position_map, descriptor, keyframe_id)
        self.landmarks[lm_id] = lm
        return lm_id

    def get_landmark(self, landmark_id: int) -> Optional[VisualLandmark]:
        return self.landmarks.get(landmark_id)

    def triangulate_from_keyframes(
        self,
        kf0: 'Keyframe',
        kf1: 'Keyframe',
        camera_matrix: np.ndarray,
        min_inliers: int = 15,
        max_reproj_error_px: float = 6.0,
    ) -> List[Tuple[int, int, int]]:
        """Triangulate matching features between kf0 and kf1 into 3D map coordinates.

        Returns:
            List of (feat_idx0, feat_idx1, landmark_id) associations.
        """
        if kf0.descriptors is None or kf1.descriptors is None or len(kf0.descriptors) == 0 or len(kf1.descriptors) == 0:
            return []

        # Feature matching between keyframes using Lowe's ratio test
        matches = self.bf_matcher.knnMatch(kf0.descriptors, kf1.descriptors, k=2)
        good_matches = []
        for m_tuple in matches:
            if len(m_tuple) == 2:
                m, n = m_tuple
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < min_inliers:
            return []

        # Relative baseline from metric keyframe poses in map frame
        dx = kf1.pose_map[0] - kf0.pose_map[0]
        dy = kf1.pose_map[1] - kf0.pose_map[1]
        baseline_m = float(np.hypot(dx, dy))
        if baseline_m < 0.05:  # Insufficient baseline for triangulation
            return []

        pts0 = np.float32([kf0.keypoints[m.queryIdx] for m in good_matches])
        pts1 = np.float32([kf1.keypoints[m.trainIdx] for m in good_matches])

        # Essential matrix estimation with RANSAC
        E, inlier_mask = cv2.findEssentialMat(
            pts0, pts1, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.5
        )
        if E is None or inlier_mask is None:
            return []

        inlier_mask = inlier_mask.ravel().astype(bool)
        if np.sum(inlier_mask) < min_inliers:
            return []

        # Relative camera pose
        _, R_rel, t_rel_unit, pose_mask = cv2.recoverPose(
            E, pts0[inlier_mask], pts1[inlier_mask], camera_matrix
        )

        # Scale relative translation by physical metric baseline
        t_rel_metric = t_rel_unit * baseline_m

        # Projection matrices: P0 = K [I | 0], P1 = K [R | t]
        P0 = camera_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P1 = camera_matrix @ np.hstack((R_rel, t_rel_metric))

        pts0_inl = pts0[inlier_mask]
        pts1_inl = pts1[inlier_mask]
        good_matches_inl = [good_matches[i] for i, inl in enumerate(inlier_mask) if inl]

        pts4d = cv2.triangulatePoints(P0, P1, pts0_inl.T, pts1_inl.T)
        pts3d_cam0 = (pts4d[:3] / np.maximum(pts4d[3:], 1e-6)).T

        associations = []
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        # Robot orientation at kf0
        th0 = kf0.pose_map[2]
        c0, s0 = np.cos(th0), np.sin(th0)

        for i, pt_cam in enumerate(pts3d_cam0):
            # Check depth plausibility along optical axis (+Z in camera optical frame)
            z_opt = pt_cam[2]
            if z_opt < 0.3 or z_opt > 25.0:
                continue

            # Reprojection check on camera 0
            u_rep0 = (fx * pt_cam[0] / z_opt) + cx
            v_rep0 = (fy * pt_cam[1] / z_opt) + cy
            rep_err0 = np.hypot(u_rep0 - pts0_inl[i, 0], v_rep0 - pts0_inl[i, 1])
            if rep_err0 > max_reproj_error_px:
                continue

            # Transform from camera optical link to base_link at kf0
            # Optical +Z is forward (+X_base), +X is right (-Y_base), +Y is down (-Z_base)
            x_base = z_opt
            y_base = -pt_cam[0]
            z_base = -pt_cam[1]

            # Transform from base_link to map frame at kf0
            x_map = kf0.pose_map[0] + (c0 * x_base - s0 * y_base)
            y_map = kf0.pose_map[1] + (s0 * x_base + c0 * y_base)
            z_map = z_base

            match = good_matches_inl[i]
            q_idx = match.queryIdx
            t_idx = match.trainIdx
            desc = kf0.descriptors[q_idx]

            lm_id = self.add_landmark((x_map, y_map, z_map), desc, kf0.keyframe_id)
            self.landmarks[lm_id].add_observation(kf1.keyframe_id, kf1.descriptors[t_idx])

            kf0.landmark_ids[q_idx] = lm_id
            kf1.landmark_ids[t_idx] = lm_id
            associations.append((q_idx, t_idx, lm_id))

        return associations

    def match_query_frame(
        self,
        query_descriptors: np.ndarray,
        ratio_thresh: float = 0.75,
    ) -> List[Tuple[int, int, float]]:
        """Match query frame descriptors against all registered map landmarks.

        Returns:
            List of (query_feature_idx, landmark_id, distance).
        """
        if query_descriptors is None or len(query_descriptors) == 0 or len(self.landmarks) == 0:
            return []

        lm_ids = list(self.landmarks.keys())
        lm_descs = np.array([self.landmarks[lid].descriptor for lid in lm_ids], dtype=np.uint8)

        matches = self.bf_matcher.knnMatch(query_descriptors, lm_descs, k=2)
        valid_matches = []
        for m_tuple in matches:
            if len(m_tuple) == 2:
                m, n = m_tuple
                if m.distance < ratio_thresh * n.distance:
                    lm_id = lm_ids[m.trainIdx]
                    valid_matches.append((m.queryIdx, lm_id, float(m.distance)))

        return valid_matches

    def to_dict(self) -> Dict[str, Any]:
        return {
            'next_id': self._next_id,
            'landmarks': [lm.to_dict() for lm in self.landmarks.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LandmarkDatabase':
        db = cls()
        db._next_id = int(data.get('next_id', 0))
        for lm_data in data.get('landmarks', []):
            lm = VisualLandmark.from_dict(lm_data)
            db.landmarks[lm.landmark_id] = lm
        return db

    def __len__(self) -> int:
        return len(self.landmarks)
