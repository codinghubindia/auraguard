"""Panorama View Path & Recovery Analyzer for NAVIGUARD UGV.

Processes wide-angle / 360-degree panoramic surround camera streams (/camera/panorama_image)
to extract ground traversability, measure obstacle distances across all azimuth angles,
detect open corridors and narrow passages (>= 0.58m), and instantaneously compute
clear recovery routes towards navigation goals without requiring blind physical rotation.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class PanoramaPathAnalyzer:
    """Analyzes panoramic surround camera frames for obstacle clearance and rapid route calculation."""

    # Vehicle geometry
    VEHICLE_WIDTH_M: float = 0.48
    NOMINAL_PASSAGE_M: float = 0.68
    TIGHT_PASSAGE_M: float = 0.58

    def __init__(
        self,
        camera_height_m: float = 0.26,
        camera_pitch_rad: float = 0.05,
        horizontal_fov_rad: float = 2.7925,  # ~160 deg or up to 2*pi for 360
        num_sectors: int = 36,
    ) -> None:
        self.camera_height_m = camera_height_m
        self.camera_pitch_rad = camera_pitch_rad
        self.horizontal_fov_rad = horizontal_fov_rad
        self.num_sectors = num_sectors

        self.last_analysis: Dict[str, Any] = {
            "panorama_available": False,
            "best_heading_deg": 0.0,
            "best_heading_rad": 0.0,
            "widest_corridor_m": 0.0,
            "clear_route_found": False,
            "escape_point": None,
            "passages": [],
            "min_obstacle_dist_m": 5.0,
        }

    def analyze_frame(
        self,
        panorama_bgr: np.ndarray,
        robot_pose: Optional[Tuple[float, float, float]] = None,
        goal_point: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Analyze a panoramic camera image for traversable sectors, obstacle distances, and clear routes.
        
        Args:
            panorama_bgr: Input BGR panoramic image (e.g. 1024x320).
            robot_pose: (x, y, yaw) of UGV in map frame.
            goal_point: (gx, gy) of navigation goal in map frame.
            
        Returns:
            Structured analysis containing best heading, passages, and escape waypoint.
        """
        if panorama_bgr is None or panorama_bgr.size == 0:
            return dict(self.last_analysis)

        h, w = panorama_bgr.shape[:2]
        gray = cv2.cvtColor(panorama_bgr, cv2.COLOR_BGR2GRAY)

        # Detect structural obstacle edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)

        # Bottom 10% is chassis/vehicle mount
        mask_y = int(h * 0.90)
        edges[mask_y:, :] = 0

        # Divide image into azimuthal angular sectors
        sector_w = max(1, w // self.num_sectors)
        passages: List[Dict[str, Any]] = []
        min_obstacle_dist = 99.0

        # Robot yaw and goal angle
        rx, ry, ryaw = robot_pose if robot_pose else (0.0, 0.0, 0.0)
        goal_yaw_rel = 0.0
        if goal_point and robot_pose:
            gx, gy = goal_point
            goal_angle_global = math.atan2(gy - ry, gx - rx)
            goal_yaw_rel = goal_angle_global - ryaw
            # Normalize to [-pi, pi]
            goal_yaw_rel = math.atan2(math.sin(goal_yaw_rel), math.cos(goal_yaw_rel))

        best_score = -1e9
        best_heading_rad = 0.0
        widest_corridor_m = 0.0
        best_passage = None

        # Effective horizon line (middle of panorama)
        horizon_y = int(h * 0.45)

        for s in range(self.num_sectors):
            col_start = s * sector_w
            col_end = min(w, (s + 1) * sector_w)

            # Azimuth angle of sector center relative to robot heading
            rel_u = (col_start + col_end) / 2.0
            norm_u = (rel_u / float(w)) - 0.5  # [-0.5 .. +0.5]
            sector_angle_rel = norm_u * self.horizontal_fov_rad

            # Look down from horizon in this sector to find closest obstacle
            sector_edges = edges[horizon_y:mask_y, col_start:col_end]
            edge_ys, edge_xs = np.nonzero(sector_edges)

            if len(edge_ys) > 0:
                # Lowest edge in image = closest obstacle on ground
                closest_edge_y = horizon_y + int(np.max(edge_ys))
                # Compute distance using ray pitch
                v_diff = max(1, closest_edge_y - horizon_y)
                pitch = math.atan2(v_diff, float(h * 0.5)) + self.camera_pitch_rad
                pitch = max(0.06, pitch)
                free_dist = self.camera_height_m / math.tan(pitch)
                free_dist = max(0.20, min(8.0, free_dist))
            else:
                # No obstacle edges in this sector -> open ground!
                free_dist = 6.0

            free_dist_bumper = max(0.05, free_dist - 0.28)
            if free_dist_bumper < min_obstacle_dist:
                min_obstacle_dist = free_dist_bumper

            # Angular width of sector
            d_theta = self.horizontal_fov_rad / float(self.num_sectors)
            # Physical width of corridor at free_dist
            corridor_w = 2.0 * free_dist * math.tan(d_theta * 1.2)
            corridor_w = max(0.30, min(3.0, corridor_w))

            widest_corridor_m = max(widest_corridor_m, corridor_w)
            is_passable = (corridor_w >= self.TIGHT_PASSAGE_M) and (free_dist_bumper >= 0.70)
            is_narrow = is_passable and (corridor_w < self.NOMINAL_PASSAGE_M)

            # Score this sector:
            # - Reward free distance ahead
            # - Reward corridor clearance width
            # - Reward alignment towards navigation goal
            alignment = math.cos(sector_angle_rel - goal_yaw_rel)
            score = (2.5 * min(free_dist_bumper, 3.5)) + (3.0 * min(corridor_w, 1.2)) + (4.0 * alignment)

            if is_passable:
                p_entry = {
                    "sector_idx": s,
                    "heading_rel_rad": round(sector_angle_rel, 3),
                    "heading_rel_deg": round(math.degrees(sector_angle_rel), 1),
                    "free_dist_m": round(free_dist_bumper, 2),
                    "corridor_width_m": round(corridor_w, 2),
                    "is_narrow": is_narrow,
                    "score": round(score, 2),
                }
                passages.append(p_entry)

                if score > best_score:
                    best_score = score
                    best_heading_rad = sector_angle_rel
                    best_passage = p_entry

        clear_route_found = (best_passage is not None)
        escape_point = None
        if clear_route_found and robot_pose:
            # Generate escape waypoint 1.8m along best heading
            esc_dist = min(2.0, max(1.2, best_passage["free_dist_m"] * 0.75))
            global_heading = ryaw + best_heading_rad
            esc_x = rx + esc_dist * math.cos(global_heading)
            esc_y = ry + esc_dist * math.sin(global_heading)
            escape_point = (round(esc_x, 2), round(esc_y, 2))

        self.last_analysis = {
            "panorama_available": True,
            "best_heading_deg": round(math.degrees(best_heading_rad), 1),
            "best_heading_rad": round(best_heading_rad, 3),
            "widest_corridor_m": round(widest_corridor_m, 2),
            "clear_route_found": clear_route_found,
            "escape_point": escape_point,
            "passages": passages,
            "min_obstacle_dist_m": round(min_obstacle_dist if min_obstacle_dist < 90.0 else 5.0, 2),
            "selected_passage": best_passage,
        }

        return dict(self.last_analysis)

    def render_panorama_overlay(
        self,
        panorama_bgr: np.ndarray,
        analysis: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render diagnostic HUD overlay on panoramic view showing open sectors and clear route."""
        canvas = panorama_bgr.copy()
        h, w = canvas.shape[:2]
        data = analysis or self.last_analysis

        best_deg = data.get("best_heading_deg", 0.0)
        best_rad = data.get("best_heading_rad", 0.0)
        passages = data.get("passages", [])

        # Highlight passable sectors in translucent green/amber
        overlay = canvas.copy()
        for p in passages:
            h_rad = p["heading_rel_rad"]
            norm_u = (h_rad / self.horizontal_fov_rad) + 0.5
            px = int(np.clip(norm_u * w, 0, w))
            sec_w = w // self.num_sectors

            col = (20, 200, 245) if p.get("is_narrow") else (40, 230, 80)
            cv2.rectangle(overlay, (px - sec_w // 2, int(h * 0.45)), (px + sec_w // 2, int(h * 0.90)), col, -1)

        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        # Highlight optimal clear route with bright arrow/beacon
        best_norm_u = (best_rad / self.horizontal_fov_rad) + 0.5
        best_px = int(np.clip(best_norm_u * w, 10, w - 10))

        # Vertical target beam
        cv2.line(canvas, (best_px, 30), (best_px, h - 20), (0, 230, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (best_px, int(h * 0.60)), 8, (0, 230, 255), -1)
        cv2.putText(canvas, f"CLEAR ROUTE: {best_deg:+.1f} deg", (best_px - 60, int(h * 0.55)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # Header banner
        cv2.rectangle(canvas, (0, 0), (w, 28), (14, 16, 22), -1)
        cv2.line(canvas, (0, 28), (w, 28), (60, 65, 75), 1)

        hud_str = (
            f"360-SURROUND PANORAMA PATH ANALYSIS | Clear Route: {best_deg:+.1f} deg | "
            f"Passages: {len(passages)} | Widest: {data.get('widest_corridor_m', 0.0):.2f}m | "
            f"Min Obs: {data.get('min_obstacle_dist_m', 5.0):.2f}m"
        )
        cv2.putText(canvas, hud_str, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (230, 230, 230), 1, cv2.LINE_AA)

        return canvas

    # Alias for convenience
    render_annotated_view = render_panorama_overlay
