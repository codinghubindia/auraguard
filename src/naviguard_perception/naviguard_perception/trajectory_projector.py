"""Dynamic Reversing-Camera Trajectory Projector and Clearance Checker.

Computes curved vehicle trajectory guidelines based on steering curvature and vehicle width,
evaluates obstacle collision and clearance along the path, and renders automotive-grade
dynamic guidelines with distance markers and PASS/BLOCKED indicators onto camera frames.
"""

import math
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np


class TrajectoryProjector:
    """Projects dynamic steering guidelines and evaluates passage clearance."""

    # Vehicle physical geometry (naviguard URDF: width=0.48m, length=0.56m)
    VEHICLE_WIDTH_M: float = 0.48
    VEHICLE_LENGTH_M: float = 0.56
    FRONT_BUMPER_X_M: float = 0.28
    INSCRIBED_RADIUS_M: float = 0.24
    SAFETY_MARGIN_M: float = 0.10
    MIN_CLEARANCE_M: float = 0.05

    # Nominal and tight passage thresholds
    NOMINAL_PASSAGE_M: float = 0.68   # 0.48 + 2 * 0.10
    TIGHT_PASSAGE_M: float = 0.58     # 0.48 + 2 * 0.05

    def __init__(
        self,
        camera_height_m: float = 0.26,
        camera_pitch_rad: float = 0.05,
        hfov_deg: float = 80.0,
    ) -> None:
        self.camera_height_m = camera_height_m
        self.camera_pitch_rad = camera_pitch_rad
        self.hfov_deg = hfov_deg

    def project_ground_to_pixel(
        self,
        x_m: float,
        y_m: float,
        img_w: int,
        img_h: int,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> Optional[Tuple[int, int]]:
        """Project ground point (x_forward, y_left) in meters to image pixel (u, v)."""
        if x_m <= 0.10:
            return None

        # Camera intrinsics fallback
        if fx is None or cx is None:
            cx = img_w / 2.0
            cy = img_h / 2.0
            fx = (img_w / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))
            fy = fx

        # Ray pitch and yaw angles
        # ray_pitch = atan2(camera_height, x_m) - pitch_downward
        ray_pitch = math.atan2(self.camera_height_m, x_m) - self.camera_pitch_rad
        if ray_pitch <= 0.01:
            ray_pitch = 0.01

        # Pixel v coordinate
        v = cy + fy * math.tan(ray_pitch)

        # Pixel u coordinate (y_m is positive left)
        # ray_yaw = atan2(y_m, x_m)
        ray_yaw = math.atan2(-y_m, x_m)
        u = cx + fx * math.tan(ray_yaw)

        # Allow slight padding outside frame for clipping
        if -50 <= u <= img_w + 50 and -50 <= v <= img_h + 50:
            return (int(round(u)), int(round(v)))
        return None

    def project_pixel_to_ground(
        self,
        u: float,
        v: float,
        img_w: int,
        img_h: int,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Convert image pixel (u, v) to ground coordinate (x_forward, y_left) in meters."""
        if cx is None or fx is None:
            cx = img_w / 2.0
            cy = img_h / 2.0
            fx = (img_w / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))
            fy = fx

        ray_yaw = math.atan2(cx - u, fx)
        ray_pitch = math.atan2(v - cy, fy) + self.camera_pitch_rad

        if ray_pitch <= 0.05:
            ray_pitch = 0.05

        x_forward = self.camera_height_m / math.tan(ray_pitch)
        x_forward = max(0.20, min(15.0, x_forward))
        y_left = x_forward * math.tan(ray_yaw)
        return float(x_forward), float(y_left)

    def generate_trajectory_points(
        self,
        cmd_vx: float,
        cmd_wz: float,
        max_dist_m: float = 3.5,
        step_m: float = 0.10,
        track_width_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute curved centerline, left rail, and right rail ground coordinates."""
        half_w = (track_width_m or self.VEHICLE_WIDTH_M) / 2.0

        # Effective curvature kappa = wz / vx
        eff_vx = cmd_vx if abs(cmd_vx) > 0.04 else 0.15
        kappa = cmd_wz / eff_vx if abs(cmd_vx) > 0.02 else (cmd_wz / 0.15)
        # Cap curvature to avoid extreme loops
        kappa = max(-2.5, min(2.5, kappa))

        num_steps = max(5, int(math.ceil(max_dist_m / step_m)))
        center_pts: List[Tuple[float, float, float]] = []  # (x, y, s)
        left_pts: List[Tuple[float, float, float]] = []
        right_pts: List[Tuple[float, float, float]] = []

        # Start slightly in front of bumper
        s_start = self.FRONT_BUMPER_X_M

        for i in range(num_steps + 1):
            s = s_start + i * step_m
            if s > max_dist_m:
                break

            if abs(kappa) < 1e-4:
                # Straight line
                xc = s
                yc = 0.0
                theta = 0.0
            else:
                # Circular arc
                radius = 1.0 / kappa
                theta = kappa * (s - s_start)
                xc = s_start + radius * math.sin(theta)
                yc = radius * (1.0 - math.cos(theta))

            # Perpendicular normal vector (left = (-sin(theta), cos(theta)))
            nx = -math.sin(theta)
            ny = math.cos(theta)

            xl = xc + half_w * nx
            yl = yc + half_w * ny

            xr = xc - half_w * nx
            yr = yc - half_w * ny

            center_pts.append((xc, yc, s))
            left_pts.append((xl, yl, s))
            right_pts.append((xr, yr, s))

        return {
            "center": center_pts,
            "left": left_pts,
            "right": right_pts,
            "curvature": kappa,
            "track_width_m": half_w * 2.0,
            "max_dist_m": max_dist_m,
        }

    def evaluate_trajectory_clearance(
        self,
        traj_points: Dict[str, Any],
        obstacles: List[Dict[str, Any]],
        collision_horizon_m: float = 1.30,
    ) -> Dict[str, Any]:
        """Check if any obstacle intersects the swept vehicle footprint along the trajectory.
        
        Returns clearance metrics and pass/fail classification.
        """
        center_pts = traj_points.get("center", [])
        half_w = traj_points.get("track_width_m", self.VEHICLE_WIDTH_M) / 2.0

        min_clearance_m = 99.0
        collision_dist_m = None
        has_collision = False
        is_tight = False
        colliding_obstacle = None

        for obs in obstacles:
            ox = float(obs.get("x_m", obs.get("x_base", 0.0)))
            oy = float(obs.get("y_m", obs.get("y_base", 0.0)))
            rad = float(obs.get("radius_m", obs.get("radius", 0.20)))

            if ox <= 0.10:
                continue

            # Check distance to all points along trajectory
            for xc, yc, s in center_pts:
                dist_to_center = math.hypot(ox - xc, oy - yc)
                clearance = dist_to_center - (half_w + rad)

                if clearance < min_clearance_m:
                    min_clearance_m = clearance

                # Direct intersection with vehicle width + minimum margin
                if clearance <= self.MIN_CLEARANCE_M:
                    if s <= collision_horizon_m:
                        has_collision = True
                        if collision_dist_m is None or s < collision_dist_m:
                            collision_dist_m = s
                            colliding_obstacle = obs
                    else:
                        is_tight = True
                elif clearance < self.SAFETY_MARGIN_M:
                    is_tight = True

        min_clearance_m = max(0.0, min_clearance_m) if min_clearance_m < 90.0 else 2.0

        if has_collision:
            status = "BLOCKED"
            can_pass = False
            reason = f"Collision with obstacle at {collision_dist_m:.2f}m"
        elif is_tight or min_clearance_m < self.SAFETY_MARGIN_M:
            status = "TIGHT"
            can_pass = True
            reason = f"Narrow passage (clearance: {min_clearance_m:.2f}m)"
        else:
            status = "SAFE"
            can_pass = True
            reason = f"Safe clear trajectory (clearance: {min_clearance_m:.2f}m)"

        return {
            "can_pass": can_pass,
            "status": status,
            "min_clearance_m": round(min_clearance_m, 2),
            "collision_dist_m": round(collision_dist_m, 2) if collision_dist_m is not None else None,
            "reason": reason,
            "colliding_obstacle": colliding_obstacle,
        }

    def render_reversing_overlay(
        self,
        canvas: np.ndarray,
        cmd_vx: float = 0.15,
        cmd_wz: float = 0.0,
        obstacles: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Render automotive-grade reversing trajectory guidelines on camera canvas."""
        h, w = canvas.shape[:2]
        meta = metadata or {}
        fx = meta.get("fx")
        fy = meta.get("fy")
        cx = meta.get("cx")
        cy = meta.get("cy")

        obs_list = obstacles or []
        traj_data = self.generate_trajectory_points(cmd_vx, cmd_wz, max_dist_m=3.2, step_m=0.08)
        clearance_info = self.evaluate_trajectory_clearance(traj_data, obs_list)

        status = clearance_info["status"]
        if status == "BLOCKED":
            main_color = (30, 30, 240)    # Bright Red
            line_thickness = 3
            status_text = "TRAJECTORY: BLOCKED [COLLISION HAZARD]"
            badge_color = (25, 25, 220)
        elif status == "TIGHT":
            main_color = (20, 200, 245)   # Amber / Yellow
            line_thickness = 2
            status_text = "TRAJECTORY: TIGHT [PROCEED WITH CAUTION]"
            badge_color = (15, 170, 220)
        else:
            main_color = (40, 240, 60)    # Bright Green
            line_thickness = 2
            status_text = "TRAJECTORY: PASS / CLEAR [SAFE TO PROCEED]"
            badge_color = (30, 180, 50)

        # Convert ground points to pixel coordinates
        pixel_left = []
        pixel_right = []
        pixel_center = []

        for p in traj_data["left"]:
            px = self.project_ground_to_pixel(p[0], p[1], w, h, fx, fy, cx, cy)
            if px and 0 <= px[1] < h:
                pixel_left.append(px)

        for p in traj_data["right"]:
            px = self.project_ground_to_pixel(p[0], p[1], w, h, fx, fy, cx, cy)
            if px and 0 <= px[1] < h:
                pixel_right.append(px)

        for p in traj_data["center"]:
            px = self.project_ground_to_pixel(p[0], p[1], w, h, fx, fy, cx, cy)
            if px and 0 <= px[1] < h:
                pixel_center.append(px)

        # 1. Fill translucent trajectory ribbon
        if len(pixel_left) >= 2 and len(pixel_right) >= 2:
            poly_pts = np.array(pixel_left + pixel_right[::-1], dtype=np.int32)
            ribbon_overlay = canvas.copy()
            cv2.fillPoly(ribbon_overlay, [poly_pts], main_color)
            alpha = 0.20 if status != "BLOCKED" else 0.32
            cv2.addWeighted(ribbon_overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

        # 2. Draw left and right curved guideline rails
        for pts_list in (pixel_left, pixel_right):
            if len(pts_list) >= 2:
                cv2.polylines(canvas, [np.array(pts_list, dtype=np.int32)], isClosed=False,
                              color=main_color, thickness=line_thickness, lineType=cv2.LINE_AA)

        # 3. Draw dashed centerline
        if len(pixel_center) >= 4:
            for i in range(0, len(pixel_center) - 1, 2):
                cv2.line(canvas, pixel_center[i], pixel_center[min(i + 1, len(pixel_center) - 1)],
                         (230, 230, 230), 1, cv2.LINE_AA)

        # 4. Draw metric distance crossbars (at 0.5m, 1.0m, 1.5m, 2.0m, 3.0m)
        dist_marks = [0.50, 1.00, 1.50, 2.00, 3.00]
        half_w = traj_data["track_width_m"] / 2.0
        kappa = traj_data["curvature"]
        s_start = self.FRONT_BUMPER_X_M

        for dm in dist_marks:
            if abs(kappa) < 1e-4:
                xc = dm
                yc = 0.0
                theta = 0.0
            else:
                radius = 1.0 / kappa
                theta = kappa * (dm - s_start)
                xc = s_start + radius * math.sin(theta)
                yc = radius * (1.0 - math.cos(theta))

            nx = -math.sin(theta)
            ny = math.cos(theta)
            p_l = self.project_ground_to_pixel(xc + half_w * nx, yc + half_w * ny, w, h, fx, fy, cx, cy)
            p_r = self.project_ground_to_pixel(xc - half_w * nx, yc - half_w * ny, w, h, fx, fy, cx, cy)

            if p_l and p_r:
                # Color code distance bar (close = red, mid = yellow, far = green)
                bar_col = (40, 40, 240) if dm <= 0.8 else ((20, 200, 245) if dm <= 1.5 else (50, 230, 80))
                cv2.line(canvas, p_l, p_r, bar_col, 2, cv2.LINE_AA)
                # Small distance tag
                mid_x = (p_l[0] + p_r[0]) // 2
                mid_y = (p_l[1] + p_r[1]) // 2
                cv2.putText(canvas, f"{dm:.1f}m", (mid_x - 14, mid_y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Trajectory Status Badge at lower HUD
        hud_box_w = 340
        hud_box_h = 24
        bx = max(10, (w - hud_box_w) // 2)
        by = h - 34

        cv2.rectangle(canvas, (bx, by), (bx + hud_box_w, by + hud_box_h), (16, 18, 24), -1)
        cv2.rectangle(canvas, (bx, by), (bx + hud_box_w, by + hud_box_h), main_color, 1)

        pass_txt = "[CAN PASS]" if clearance_info["can_pass"] else "[BLOCKED]"
        cv2.putText(canvas, pass_txt, (bx + 8, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, main_color, 1, cv2.LINE_AA)

        clr_txt = f"Clearance: {clearance_info['min_clearance_m']:.2f}m"
        cv2.putText(canvas, clr_txt, (bx + 110, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

        stat_txt = f"{status}"
        cv2.putText(canvas, stat_txt, (bx + 265, by + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, main_color, 1, cv2.LINE_AA)

        return canvas, clearance_info
