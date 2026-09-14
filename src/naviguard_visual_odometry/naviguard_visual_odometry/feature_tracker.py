"""Modular Feature Detection, Optical Flow Tracking, and Visual Motion Measurement.

Implements Shi-Tomasi corner detection, Lucas-Kanade pyramidal optical flow
with bidirectional (forward-backward) error validation, robust outlier rejection,
and visual displacement statistics.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import cv2
import numpy as np


class FeatureTrackerConfig:
    """Configuration parameters for feature detection and optical flow tracking."""

    def __init__(
        self,
        max_features: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 12.0,
        block_size: int = 3,
        use_harris: bool = False,
        lk_win_size: Tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
        lk_max_iters: int = 30,
        lk_eps: float = 0.01,
        fb_err_threshold: float = 1.0,
        min_features_threshold: int = 70,
        outlier_mad_k: float = 3.5,
        max_displacement_px: float = 80.0,
        self_mask_height_ratio: float = 0.15,
    ) -> None:
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.use_harris = use_harris
        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level
        self.lk_max_iters = lk_max_iters
        self.lk_eps = lk_eps
        self.fb_err_threshold = fb_err_threshold
        self.min_features_threshold = min_features_threshold
        self.outlier_mad_k = outlier_mad_k
        self.max_displacement_px = max_displacement_px
        self.self_mask_height_ratio = self_mask_height_ratio


class TrackedFeature:
    """Represents an individual tracked visual feature between consecutive frames."""

    def __init__(
        self,
        prev_pt: Tuple[float, float],
        curr_pt: Tuple[float, float],
        is_inlier: bool = True,
    ) -> None:
        self.prev_x, self.prev_y = prev_pt
        self.curr_x, self.curr_y = curr_pt
        self.dx = self.curr_x - self.prev_x
        self.dy = self.curr_y - self.prev_y
        self.displacement = float(np.hypot(self.dx, self.dy))
        self.is_inlier = is_inlier


class FeatureTracker:
    """Core visual motion tracking engine using Shi-Tomasi and Lucas-Kanade."""

    def __init__(self, config: Optional[FeatureTrackerConfig] = None) -> None:
        self.config = config or FeatureTrackerConfig()
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None  # Shape (N, 1, 2)
        self.last_status_message = "INITIALIZING"
        self.trajectory_history: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.accumulated_distance_m: float = 0.0
        self.cur_x_m: float = 0.0
        self.cur_y_m: float = 0.0
        self.cur_yaw_deg: float = 0.0

    def detect_features(
        self,
        gray: np.ndarray,
        existing_pts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Detect strong Shi-Tomasi corners in grayscale image, avoiding existing feature points.

        Applies a self-robot exclusion mask to prevent feature detection on the vehicle bumper/hood.
        """
        needed = self.config.max_features
        h, w = gray.shape[:2]

        mask = None
        has_existing = existing_pts is not None and len(existing_pts) > 0

        if self.config.self_mask_height_ratio > 0.0 or has_existing:
            mask = np.full((h, w), 255, dtype=np.uint8)

            # Mask out self-robot region (lower portion of the frame)
            if self.config.self_mask_height_ratio > 0.0:
                mask_y = int(h * (1.0 - self.config.self_mask_height_ratio))
                mask[mask_y:, :] = 0

            # Mask out existing points
            if has_existing:
                needed = self.config.max_features - len(existing_pts)
                if needed <= 0:
                    return existing_pts
                for pt in existing_pts:
                    px, py = int(pt[0][0]), int(pt[0][1])
                    cv2.circle(mask, (px, py), int(self.config.min_distance), 0, -1)

        new_corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=needed,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
            useHarrisDetector=self.config.use_harris,
            mask=mask,
        )

        if new_corners is None or len(new_corners) == 0:
            return existing_pts if existing_pts is not None else np.empty((0, 1, 2), dtype=np.float32)

        if existing_pts is not None and len(existing_pts) > 0:
            return np.vstack([existing_pts, new_corners.astype(np.float32)])
        return new_corners.astype(np.float32)

    def track(
        self,
        curr_gray: np.ndarray,
    ) -> Tuple[List[TrackedFeature], Dict[str, Any]]:
        """Track features from previous frame to current frame and calculate motion statistics.

        Returns:
            Tuple of (tracked_features_list, visual_motion_statistics_dict).
        """
        h, w = curr_gray.shape

        # Initial frame handling
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_gray = curr_gray.copy()
            self.prev_pts = self.detect_features(curr_gray)
            self.last_status_message = "FIRST_FRAME_DETECTED"
            return [], self._empty_stats(len(self.prev_pts), "FIRST_FRAME")

        # Insufficient features from previous frame -> replenish before tracking
        if len(self.prev_pts) < self.config.min_features_threshold:
            self.prev_pts = self.detect_features(self.prev_gray, self.prev_pts)
            self.last_status_message = "REPLENISHED_PREV_FEATURES"

        if len(self.prev_pts) == 0:
            self.prev_gray = curr_gray.copy()
            self.prev_pts = self.detect_features(curr_gray)
            self.last_status_message = "NO_FEATURES_AVAILABLE"
            return [], self._empty_stats(len(self.prev_pts), "NO_FEATURES")

        lk_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            self.config.lk_max_iters,
            self.config.lk_eps,
        )

        # 1. Forward Optical Flow: prev -> curr
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.prev_pts,
            None,
            winSize=self.config.lk_win_size,
            maxLevel=self.config.lk_max_level,
            criteria=lk_criteria,
        )

        # 2. Backward Optical Flow: curr -> prev (Bidirectional validation)
        p0_back, st0, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray,
            self.prev_gray,
            p1,
            None,
            winSize=self.config.lk_win_size,
            maxLevel=self.config.lk_max_level,
            criteria=lk_criteria,
        )

        # 3. Filter valid bidirectional tracks
        raw_tracks: List[TrackedFeature] = []
        valid_curr_pts: List[np.ndarray] = []

        num_input_features = len(self.prev_pts)
        for i in range(len(self.prev_pts)):
            if st1[i][0] != 1 or st0[i][0] != 1:
                continue

            pt0 = self.prev_pts[i][0]
            pt1 = p1[i][0]
            pt0_b = p0_back[i][0]

            # Forward-backward distance check
            fb_err = float(np.hypot(pt0[0] - pt0_b[0], pt0[1] - pt0_b[1]))
            if fb_err > self.config.fb_err_threshold:
                continue

            # Check inside frame boundary
            if not (0 <= pt1[0] < w and 0 <= pt1[1] < h):
                continue

            track = TrackedFeature((pt0[0], pt0[1]), (pt1[0], pt1[1]))
            raw_tracks.append(track)
            valid_curr_pts.append(p1[i])

        # 4. Outlier rejection based on robust displacement statistics
        inlier_tracks: List[TrackedFeature] = []
        if len(raw_tracks) > 0:
            displacements = np.array([t.displacement for t in raw_tracks], dtype=np.float32)
            med_disp = float(np.median(displacements))
            abs_devs = np.abs(displacements - med_disp)
            mad = float(np.median(abs_devs))
            spread_thresh = max(self.config.outlier_mad_k * max(mad, 1.0), 3.0)

            for t in raw_tracks:
                if (
                    abs(t.displacement - med_disp) <= spread_thresh
                    and t.displacement <= self.config.max_displacement_px
                ):
                    t.is_inlier = True
                    inlier_tracks.append(t)
                else:
                    t.is_inlier = False

        num_tracked = len(raw_tracks)
        num_inliers = len(inlier_tracks)
        tracking_ratio = float(num_inliers) / float(max(num_input_features, 1))

        # 5. Compute Visual Motion Statistics across inliers
        if num_inliers > 0:
            dx_arr = np.array([t.dx for t in inlier_tracks], dtype=np.float32)
            dy_arr = np.array([t.dy for t in inlier_tracks], dtype=np.float32)
            disp_arr = np.array([t.displacement for t in inlier_tracks], dtype=np.float32)

            stats = {
                'num_detected': num_input_features,
                'num_tracked': num_tracked,
                'num_inliers': num_inliers,
                'tracking_ratio': tracking_ratio,
                'mean_dx_px': float(np.mean(dx_arr)),
                'mean_dy_px': float(np.mean(dy_arr)),
                'median_dx_px': float(np.median(dx_arr)),
                'median_dy_px': float(np.median(dy_arr)),
                'mean_displacement_px': float(np.mean(disp_arr)),
                'median_displacement_px': float(np.median(disp_arr)),
                'disp_std_px': float(np.std(disp_arr)),
                'status': 'TRACKING_OK' if tracking_ratio >= 0.3 else 'TRACKING_DEGRADED',
                'is_valid': True,
            }
        else:
            stats = self._empty_stats(num_input_features, 'TRACKING_LOST')

        # 6. Prepare state for next frame: replenish if inlier count is low
        inlier_curr_pts = [
            p1[i] for i, t in enumerate(raw_tracks) if t.is_inlier
        ]
        if len(inlier_curr_pts) > 0:
            next_pts = np.array(inlier_curr_pts, dtype=np.float32).reshape(-1, 1, 2)
        else:
            next_pts = np.empty((0, 1, 2), dtype=np.float32)

        if len(next_pts) < self.config.min_features_threshold:
            next_pts = self.detect_features(curr_gray, next_pts)
            stats['status'] = f"{stats['status']}_REDETECTED"

        self.prev_gray = curr_gray.copy()
        self.prev_pts = next_pts
        self.last_status_message = stats['status']

        return raw_tracks, stats

    def _empty_stats(self, num_detected: int, status: str) -> Dict[str, Any]:
        """Generate zeroed visual motion statistics for initialization or tracking failure."""
        return {
            'num_detected': num_detected,
            'num_tracked': 0,
            'num_inliers': 0,
            'tracking_ratio': 0.0,
            'mean_dx_px': 0.0,
            'mean_dy_px': 0.0,
            'median_dx_px': 0.0,
            'median_dy_px': 0.0,
            'mean_displacement_px': 0.0,
            'median_displacement_px': 0.0,
            'disp_std_px': 0.0,
            'status': status,
            'is_valid': False,
        }

    def render_debug_canvas(
        self,
        curr_bgr: np.ndarray,
        tracks: List[TrackedFeature],
        stats: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Render optical flow motion vector overlay, BEV visual trajectory, and real-world telemetry HUD."""
        h, w = curr_bgr.shape[:2]
        canvas = curr_bgr.copy()
        meta = metadata or {}

        # 0. Update internal visual trajectory accumulator
        dt_sim = float(meta.get('dt_sim', 0.05))
        if dt_sim <= 0.0 or dt_sim > 0.5:
            dt_sim = 0.05
        wheel_vx = meta.get('wheel_vx')
        wheel_wz = meta.get('wheel_wz')
        geom_valid = bool(meta.get('geom_is_valid', False))

        # Heading & displacement integration
        if geom_valid:
            dyaw = math.radians(float(meta.get('rot_yaw_deg', 0.0)))
        elif wheel_wz is not None:
            dyaw = float(wheel_wz) * dt_sim
        else:
            dyaw = -float(stats.get('median_dx_px', 0.0)) * 0.002

        self.cur_yaw_deg += math.degrees(dyaw)
        self.cur_yaw_deg = (self.cur_yaw_deg + 180.0) % 360.0 - 180.0

        if wheel_vx is not None:
            ds = float(wheel_vx) * dt_sim
        else:
            ds = max(0.0, float(stats.get('median_dy_px', 0.0)) * 0.015)

        rad = math.radians(self.cur_yaw_deg)
        self.cur_x_m += ds * math.cos(rad)
        self.cur_y_m += ds * math.sin(rad)
        self.accumulated_distance_m += abs(ds)

        if len(self.trajectory_history) == 0 or math.hypot(self.cur_x_m - self.trajectory_history[-1][0], self.cur_y_m - self.trajectory_history[-1][1]) > 0.04:
            self.trajectory_history.append((self.cur_x_m, self.cur_y_m))
            if len(self.trajectory_history) > 150:
                self.trajectory_history.pop(0)

        # 1. Optical Center, Horizon, and Focus of Expansion (FOE)
        foe_x = w // 2
        foe_y = int(h * 0.44)
        cv2.line(canvas, (0, foe_y), (w, foe_y), (40, 75, 55), 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, (foe_x, foe_y), (0, 220, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1)
        cv2.circle(canvas, (foe_x, foe_y), 5, (0, 220, 255), 1, cv2.LINE_AA)

        # 2. Draw real-world target reticles and optical flow motion vectors
        for t in tracks:
            p0 = (int(round(t.prev_x)), int(round(t.prev_y)))
            p1 = (int(round(t.curr_x)), int(round(t.curr_y)))

            if t.is_inlier:
                if t.displacement < 2.0:
                    pt_col = (0, 255, 120)    # Neon green (stable ground/feature)
                elif t.displacement < 5.0:
                    pt_col = (255, 230, 70)   # Electric cyan (active flow)
                else:
                    pt_col = (0, 215, 255)    # Amber gold (rapid movement)

                cv2.circle(canvas, p1, 3, pt_col, 1, cv2.LINE_AA)
                cv2.circle(canvas, p1, 1, pt_col, -1, cv2.LINE_AA)

                if t.displacement >= 1.2:
                    cv2.arrowedLine(canvas, p0, p1, pt_col, 1, tipLength=0.28, line_type=cv2.LINE_AA)
            else:
                cv2.drawMarker(canvas, p1, (40, 40, 220), markerType=cv2.MARKER_TILTED_CROSS, markerSize=5, thickness=1)

        # 3. Dedicated Real-Time 2D BEV Visual Trajectory Minimap (Bottom-Right Corner)
        inset_w, inset_h = 136, 106
        inset_x = w - inset_w - 8
        inset_y = h - 50 - inset_h - 4
        cv2.rectangle(canvas, (inset_x, inset_y), (inset_x + inset_w, inset_y + inset_h), (14, 16, 22), -1)
        cv2.rectangle(canvas, (inset_x, inset_y), (inset_x + inset_w, inset_y + inset_h), (50, 180, 220), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "VO BEV TRAIL", (inset_x + 6, inset_y + 13), font, 0.32, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"S: {self.accumulated_distance_m:.1f}m", (inset_x + inset_w - 48, inset_y + 13), font, 0.30, (80, 255, 80), 1, cv2.LINE_AA)

        map_cx = inset_x + inset_w // 2
        map_cy = inset_y + inset_h // 2 + 8
        cv2.drawMarker(canvas, (map_cx, map_cy), (60, 70, 80), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)
        cv2.circle(canvas, (map_cx, map_cy), 18, (30, 36, 44), 1, cv2.LINE_AA)
        cv2.circle(canvas, (map_cx, map_cy), 36, (30, 36, 44), 1, cv2.LINE_AA)

        scale_ppm = 7.0
        if len(self.trajectory_history) >= 2:
            pts_bev = []
            for tx, ty in self.trajectory_history:
                rx = tx - self.cur_x_m
                ry = ty - self.cur_y_m
                fwd = rx * math.cos(rad) + ry * math.sin(rad)
                lat = -rx * math.sin(rad) + ry * math.cos(rad)
                px = int(map_cx + lat * scale_ppm)
                py = int(map_cy - fwd * scale_ppm)
                if inset_x + 2 <= px <= inset_x + inset_w - 2 and inset_y + 16 <= py <= inset_y + inset_h - 2:
                    pts_bev.append((px, py))
            if len(pts_bev) >= 2:
                for i in range(len(pts_bev) - 1):
                    cv2.line(canvas, pts_bev[i], pts_bev[i + 1], (255, 200, 40), 1, cv2.LINE_AA)

        arrow_tip = (map_cx, map_cy - 7)
        cv2.line(canvas, (map_cx, map_cy + 4), arrow_tip, (0, 255, 120), 2, cv2.LINE_AA)
        cv2.circle(canvas, (map_cx, map_cy), 2, (0, 255, 120), -1)

        # 4. Render Real-World Production HUD Telemetry Banners
        scale_small = 0.38
        scale_title = 0.44
        c_white = (240, 240, 240)
        c_yellow = (0, 230, 255)
        c_green = (80, 255, 80)
        c_cyan = (255, 230, 100)
        c_orange = (0, 180, 255)
        c_magenta = (255, 120, 220)

        # Top banner
        top_h = 46
        cv2.rectangle(canvas, (0, 0), (w, top_h), (18, 18, 22), -1)
        cv2.line(canvas, (0, top_h), (w, top_h), (70, 70, 80), 1)

        frame_idx = meta.get('frame_idx', 0)
        stamp_sec = meta.get('stamp_sec', 0)
        stamp_nanosec = meta.get('stamp_nanosec', 0)
        frame_id = meta.get('frame_id', 'camera_link')
        short_frame_id = frame_id.split('/')[-1] if '/' in frame_id else frame_id

        title_text = f"NAVIGUARD VISUAL ODOMETRY | Frame #{frame_idx}"
        cv2.putText(canvas, title_text, (8, 18), font, scale_title, c_yellow, 1, cv2.LINE_AA)

        hdg_text = f"[HDG: {self.cur_yaw_deg:+.1f}°]"
        h_sz, _ = cv2.getTextSize(hdg_text, font, scale_small, 1)
        cv2.putText(canvas, hdg_text, (w // 2 - h_sz[0] // 2, 18), font, scale_small, (255, 230, 100), 1, cv2.LINE_AA)

        input_fps = meta.get('input_fps', 0.0)
        latency_ms = meta.get('proc_latency_ms', 0.0)
        proc_fps = meta.get('proc_fps', 0.0)
        perf_text = f"In: {input_fps:.1f} FPS | Lat: {latency_ms:.1f}ms | Cap: {proc_fps:.0f} FPS"
        t_sz, _ = cv2.getTextSize(perf_text, font, scale_small, 1)
        cv2.putText(canvas, perf_text, (max(w - t_sz[0] - 8, 300), 18), font, scale_small, c_green, 1, cv2.LINE_AA)

        # Line 2: Feature Tracking Health Bar
        n_inliers = stats.get('num_inliers', 0)
        n_det = stats.get('num_detected', 0)
        ratio = stats.get('tracking_ratio', 0.0) * 100.0
        status_str = stats.get('status', 'OK')

        bar_len = 10
        fill_bars = int(round((ratio / 100.0) * bar_len))
        fill_bars = max(0, min(bar_len, fill_bars))
        bar_str = "[" + "=" * fill_bars + "-" * (bar_len - fill_bars) + "]"

        feat_text = f"Tracks: {n_inliers}/{n_det} ({ratio:.1f}%) {bar_str} | Status: {status_str}"
        cv2.putText(canvas, feat_text, (8, 37), font, scale_small, c_white, 1, cv2.LINE_AA)

        sim_str = f"Sim: {stamp_sec}.{stamp_nanosec // 1000000:03d}s | {short_frame_id}"
        s_sz, _ = cv2.getTextSize(sim_str, font, scale_small, 1)
        cv2.putText(canvas, sim_str, (max(w - s_sz[0] - 8, 300), 37), font, scale_small, c_cyan, 1, cv2.LINE_AA)

        # Bottom banner (48px)
        bot_h = 48
        cv2.rectangle(canvas, (0, h - bot_h), (w, h), (18, 18, 22), -1)
        cv2.line(canvas, (0, h - bot_h), (w, h - bot_h), (70, 70, 80), 1)

        med_dx = stats.get('median_dx_px', 0.0)
        med_dy = stats.get('median_dy_px', 0.0)
        med_disp = stats.get('median_displacement_px', 0.0)
        disp_std = stats.get('disp_std_px', 0.0)
        motion_text = f"Visual Flow: dx={med_dx:+.2f} dy={med_dy:+.2f} px | Disp={med_disp:.2f}+/-{disp_std:.2f} px"
        cv2.putText(canvas, motion_text, (8, h - 28), font, scale_small, c_orange, 1, cv2.LINE_AA)

        if wheel_vx is not None and wheel_wz is not None:
            odom_text = f"Est Vel: vx={wheel_vx:+.2f}m/s wz={wheel_wz:+.2f}r/s | S_vo={self.accumulated_distance_m:.2f}m"
        else:
            odom_text = f"Est Vel: S_vo={self.accumulated_distance_m:.2f}m (/odom standby)"
        o_sz, _ = cv2.getTextSize(odom_text, font, scale_small, 1)
        cv2.putText(canvas, odom_text, (max(w - o_sz[0] - 8, 280), h - 28), font, scale_small, (150, 220, 150), 1, cv2.LINE_AA)

        geom_status = meta.get('geom_status', 'WAITING_CALIB')
        scale_status = meta.get('translation_scale_status', 'UNKNOWN')

        if geom_valid:
            yaw_d = meta.get('rot_yaw_deg', 0.0)
            u_tx = meta.get('unit_tx', 0.0)
            u_ty = meta.get('unit_ty', 0.0)
            u_tz = meta.get('unit_tz', 0.0)
            g_inl = meta.get('geom_num_inliers', 0)
            geom_text = f"Geom 6-DOF: OK ({g_inl} inl) | Yaw: {yaw_d:+.2f}° | t_dir: [{u_tx:+.2f}, {u_ty:+.2f}, {u_tz:+.2f}] | Scale: {scale_status}"
            cv2.putText(canvas, geom_text, (8, h - 10), font, scale_small, c_green, 1, cv2.LINE_AA)
        else:
            geom_text = f"Geom 6-DOF: {geom_status} | Scale: {scale_status} | Traversed: {self.accumulated_distance_m:.2f}m"
            cv2.putText(canvas, geom_text, (8, h - 10), font, scale_small, c_magenta, 1, cv2.LINE_AA)

        return canvas
