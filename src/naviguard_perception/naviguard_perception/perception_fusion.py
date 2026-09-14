"""NAVIGUARD Perception Fusion and Temporal Consistency Module.

Fuses:
1. Classical OpenCV edge & corridor traversability
2. Learned YOLO semantic object detections
3. Ground-plane geometric projections
4. Multi-frame temporal persistence tracking
5. Confidence layer outputs (visual, object, traversability)
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np


@dataclass
class TrackedObstacle:
    """Represents a spatially tracked obstacle candidate over time."""
    track_id: int
    class_name: str
    confidence: float
    bbox: List[int]                     # [x, y, w, h] in image
    ground_pos_m: Tuple[float, float]   # (x_forward, y_left) relative to base_link
    first_seen_ts: float
    last_seen_ts: float
    hit_count: int = 1                  # Consecutive frame observations
    miss_count: int = 0
    is_confirmed: bool = False          # Promoted after >= min_persistence_frames
    emergency: bool = False

    @property
    def hits(self) -> int:
        return self.hit_count

    @property
    def confirmed(self) -> bool:
        return self.is_confirmed

    @property
    def x_base(self) -> float:
        return self.ground_pos_m[0]

    @property
    def y_base(self) -> float:
        return self.ground_pos_m[1]


FusedObstacle = TrackedObstacle


class PerceptionFusion:
    """Integrates classical visual evidence with YOLO semantics and temporal consistency."""

    def __init__(
        self,
        min_persistence_frames: int = 3,
        max_miss_frames: int = 5,
        max_association_dist_m: float = 0.50,
        camera_height_m: float = 0.26,
        camera_pitch_rad: float = 0.05,  # Slight downward tilt
        camera_hfov_deg: float = 80.0,
        min_hits_to_confirm: Optional[int] = None,
        max_misses_to_drop: Optional[int] = None,
    ) -> None:
        self.min_persistence_frames = min_hits_to_confirm if min_hits_to_confirm is not None else min_persistence_frames
        self.max_miss_frames = max_misses_to_drop if max_misses_to_drop is not None else max_miss_frames
        self.max_association_dist_m = max_association_dist_m
        self.camera_height_m = camera_height_m
        self.camera_pitch_rad = camera_pitch_rad
        self.camera_hfov_deg = camera_hfov_deg

        self.tracked_obstacles: Dict[int, TrackedObstacle] = {}
        self._next_track_id: int = 1

        # Confidence metrics
        self.visual_confidence: float = 1.0
        self.object_detection_confidence: float = 1.0
        self.traversability_confidence: float = 1.0

    def get_tracked_obstacles(self) -> List[TrackedObstacle]:
        """Return list of current obstacle tracks."""
        return list(self.tracked_obstacles.values())

    def update(
        self,
        bgr_image: Optional[np.ndarray] = None,
        classical_traversability_mask: Optional[np.ndarray] = None,
        yolo_detections: Optional[List[Dict[str, Any]]] = None,
        timestamp: float = 0.0,
        detections: Optional[List[Dict[str, Any]]] = None,
        classical_contours: Optional[Any] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
        """Fuse classical segmentation with YOLO detections, tracking obstacles over time.
        
        Returns:
            fused_overlay (np.ndarray): Diagnostic RGB image.
            confirmed_obstacles (list): Confirmed spatial obstacles with ground coordinates (x, y, radius).
            confidence_scores (dict): Dictionary of visual confidences.
        """
        now = timestamp if timestamp > 0.0 else time.time()
        if bgr_image is None:
            bgr_image = np.zeros((480, 640, 3), dtype=np.uint8)
        if classical_traversability_mask is None:
            classical_traversability_mask = np.full((480, 640), 255, dtype=np.uint8)
        if yolo_detections is None:
            yolo_detections = detections if detections is not None else []

        h, w = bgr_image.shape[:2]

        # 1. Project detections onto ground plane relative to robot base_link
        current_detections = []
        for det in yolo_detections:
            bx, by, bw, bh = det.get("bounding_box", det.get("box", [0, 0, 10, 10]))
            # Bottom center of bounding box represents contact with ground
            u_ground = bx + bw / 2.0
            v_ground = by + bh
            gx, gy = self._project_pixel_to_ground(u_ground, v_ground, w, h)

            current_detections.append({
                "detection": det,
                "ground_pos_m": (gx, gy),
            })

        # 2. Update multi-frame temporal obstacle tracker
        self._update_tracks(current_detections, now)

        # 3. Generate confirmed obstacles for navigation costmap
        confirmed_obstacles: List[Dict[str, Any]] = []
        for track in self.tracked_obstacles.values():
            if track.is_confirmed and track.ground_pos_m[0] > 0.10:
                # Estimate physical radius from bounding box width at distance
                dist = math.hypot(track.ground_pos_m[0], track.ground_pos_m[1])
                # Conservative radius between 0.15m and 0.45m
                radius_m = max(0.15, min(0.45, dist * 0.10))
                confirmed_obstacles.append({
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "confidence": track.confidence,
                    "x_m": round(track.ground_pos_m[0], 2),
                    "y_m": round(track.ground_pos_m[1], 2),
                    "radius_m": round(radius_m, 2),
                    "is_lethal": True,
                })

        # 4. Calculate perception confidences
        self._calculate_confidences(bgr_image, yolo_detections, classical_traversability_mask)

        # 5. Build diagnostic fused overlay
        fused_overlay = self._render_fused_overlay(
            bgr_image,
            classical_traversability_mask,
            self.tracked_obstacles,
        )

        confidences = {
            "visual_confidence": self.visual_confidence,
            "object_detection_confidence": self.object_detection_confidence,
            "traversability_confidence": self.traversability_confidence,
        }

        return fused_overlay, confirmed_obstacles, confidences

    def _project_pixel_to_ground(self, u: float, v: float, img_w: int, img_h: int) -> Tuple[float, float]:
        """Convert pixel coordinates (u, v) to ground coordinate (x_forward, y_left) in meters."""
        # Principal point and focal length approximation from FOV
        cx = img_w / 2.0
        cy = img_h / 2.0
        fx = (img_w / 2.0) / math.tan(math.radians(self.camera_hfov_deg / 2.0))
        fy = fx

        # Pixel ray angles
        ray_yaw = math.atan2(cx - u, fx)
        ray_pitch = math.atan2(v - cy, fy) + self.camera_pitch_rad

        # Avoid rays parallel to or above horizon
        if ray_pitch <= 0.05:
            ray_pitch = 0.05

        # Ground intersection distance
        x_forward = self.camera_height_m / math.tan(ray_pitch)
        x_forward = max(0.20, min(15.0, x_forward))
        y_left = x_forward * math.tan(ray_yaw)

        return x_forward, y_left

    def _update_tracks(self, current_detections: List[Dict[str, Any]], now: float) -> None:
        """Associate new detections with existing tracks using nearest-neighbor centroid matching."""
        assigned_tracks = set()

        for cur in current_detections:
            gx, gy = cur["ground_pos_m"]
            det = cur["detection"]

            # Find closest existing track
            best_track_id = None
            min_dist = float('inf')

            for tid, track in self.tracked_obstacles.items():
                if tid in assigned_tracks:
                    continue
                d = math.hypot(gx - track.ground_pos_m[0], gy - track.ground_pos_m[1])
                if d < min_dist and d <= self.max_association_dist_m:
                    min_dist = d
                    best_track_id = tid

            if best_track_id is not None:
                # Update existing track
                t = self.tracked_obstacles[best_track_id]
                t.bbox = det.get("bounding_box", det.get("box", [0, 0, 10, 10]))
                t.confidence = 0.7 * t.confidence + 0.3 * det["confidence"]
                t.ground_pos_m = (0.6 * t.ground_pos_m[0] + 0.4 * gx, 0.6 * t.ground_pos_m[1] + 0.4 * gy)
                t.last_seen_ts = now
                t.hit_count += 1
                t.miss_count = 0

                # Promote to confirmed if persistent
                if t.hit_count >= self.min_persistence_frames:
                    t.is_confirmed = True
                assigned_tracks.add(best_track_id)
            else:
                # Create new track candidate
                # Emergency near-field bypass: if object is right in front (<1.0m) with high confidence, confirm immediately
                dist_m = math.hypot(gx, gy)
                immediate_confirm = (dist_m < 1.0 and det["confidence"] >= 0.85)

                new_track = TrackedObstacle(
                    track_id=self._next_track_id,
                    class_name=det["class_name"],
                    confidence=det["confidence"],
                    bbox=det.get("bounding_box", det.get("box", [0, 0, 10, 10])),
                    ground_pos_m=(gx, gy),
                    first_seen_ts=now,
                    last_seen_ts=now,
                    hit_count=1,
                    miss_count=0,
                    is_confirmed=immediate_confirm,
                    emergency=immediate_confirm,
                )
                self.tracked_obstacles[self._next_track_id] = new_track
                self._next_track_id += 1

        # Handle misses and age-out
        expired_ids = []
        for tid, track in self.tracked_obstacles.items():
            if tid not in assigned_tracks:
                track.miss_count += 1
                if track.miss_count > self.max_miss_frames or (now - track.last_seen_ts) > 1.0:
                    expired_ids.append(tid)

        for tid in expired_ids:
            del self.tracked_obstacles[tid]

    def _calculate_confidences(
        self,
        bgr_image: np.ndarray,
        yolo_detections: List[Dict[str, Any]],
        classical_mask: np.ndarray,
    ) -> None:
        """Evaluate scene illumination, sharpness, and segmentation quality."""
        # 1. Visual confidence (illumination and blur check)
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = float(np.mean(gray))

        # Sharpness score (variance of Laplacian): normal sharp outdoor scene > 80
        sharpness_score = min(1.0, max(0.2, laplacian_var / 120.0))
        # Brightness score: optimal 40 - 210
        if 40 <= mean_brightness <= 210:
            bright_score = 1.0
        elif mean_brightness < 40:
            bright_score = max(0.2, mean_brightness / 40.0)
        else:
            bright_score = max(0.3, (255.0 - mean_brightness) / 45.0)

        self.visual_confidence = round(0.5 * sharpness_score + 0.5 * bright_score, 3)

        # 2. Object detection confidence
        if yolo_detections:
            avg_conf = float(np.mean([d["confidence"] for d in yolo_detections]))
            self.object_detection_confidence = round(avg_conf, 3)
        else:
            self.object_detection_confidence = 1.0  # Clear scene

        # 3. Traversability confidence (consistency between classical free space and absence of lethal obstacles)
        if classical_mask is not None and classical_mask.size > 0:
            free_ratio = float(np.count_nonzero(classical_mask > 128)) / float(classical_mask.size)
            self.traversability_confidence = round(min(1.0, 0.4 + free_ratio * 0.6), 3)
        else:
            self.traversability_confidence = 0.85

    def _render_fused_overlay(
        self,
        bgr_image: np.ndarray,
        classical_mask: np.ndarray,
        tracks: Dict[int, TrackedObstacle],
    ) -> np.ndarray:
        """Blend classical traversability green/amber tints with YOLO track boxes."""
        vis = bgr_image.copy()

        # Overlay classical traversability corridor tint in lower 50% of frame
        if classical_mask is not None and classical_mask.shape[:2] == bgr_image.shape[:2]:
            green_tint = np.zeros_like(vis)
            green_tint[:, :] = (0, 180, 0)
            free_indices = classical_mask > 128
            vis[free_indices] = cv2.addWeighted(vis[free_indices], 0.70, green_tint[free_indices], 0.30, 0)

        # Overlay tracked obstacles
        for track in tracks.values():
            x, y, w, h = track.bbox
            is_conf = track.is_confirmed

            # Color: Solid Red for CONFIRMED, Dashed Amber for CANDIDATE
            col = (0, 0, 255) if is_conf else (0, 200, 255)
            thickness = 2 if is_conf else 1

            cv2.rectangle(vis, (x, y), (x + w, y + h), col, thickness)

            # Ground position tag
            tag_status = "CONFIRMED" if is_conf else f"CANDIDATE ({track.hit_count}/3)"
            gx, gy = track.ground_pos_m
            dist_str = f"{tag_status} | {track.class_name.upper()} ({gx:.1f}m, {gy:+.1f}m)"

            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(dist_str, font, 0.40, 1)
            cv2.rectangle(vis, (x, max(0, y - th - 4)), (x + tw + 4, max(0, y)), col, -1)
            cv2.putText(vis, dist_str, (x + 2, max(th + 1, y - 2)), font, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        # Summary Header
        conf_summary = f"FUSION: Vis {int(self.visual_confidence*100)}% | Trav {int(self.traversability_confidence*100)}% | Tracks: {len(tracks)}"
        cv2.putText(vis, conf_summary, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        return vis
