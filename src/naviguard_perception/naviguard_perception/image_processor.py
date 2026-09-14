"""Modular Image Processing and Visual Diagnostic Pipeline for NAVIGUARD UGV.

Provides outdoor visual enhancement, structural edge detection,
ground traversability region indication, and diagnostic telemetry overlays.
Designed as a modular preprocessing foundation for future AI perception,
segmentation, and visual localization models.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from naviguard_perception.trajectory_projector import TrajectoryProjector


class ImageProcessorConfig:
    """Configuration parameters for the perception image processor."""

    def __init__(
        self,
        display_mode: str = 'overlay',
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        canny_low_threshold: int = 50,
        canny_high_threshold: int = 150,
        gaussian_blur_ksize: int = 5,
        ground_roi_top_ratio: float = 0.60,
        ground_roi_bottom_ratio: float = 0.98,
        ground_roi_top_width_ratio: float = 0.50,
        ground_roi_bottom_width_ratio: float = 0.90,
        self_mask_height_ratio: float = 0.15,
    ) -> None:
        self.display_mode = display_mode  # 'overlay' or 'side_by_side'
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.canny_low_threshold = canny_low_threshold
        self.canny_high_threshold = canny_high_threshold
        self.gaussian_blur_ksize = gaussian_blur_ksize
        self.ground_roi_top_ratio = ground_roi_top_ratio
        self.ground_roi_bottom_ratio = ground_roi_bottom_ratio
        self.ground_roi_top_width_ratio = ground_roi_top_width_ratio
        self.ground_roi_bottom_width_ratio = ground_roi_bottom_width_ratio
        self.self_mask_height_ratio = self_mask_height_ratio


class ImageProcessor:
    """Core image processing class for NAVIGUARD perception."""

    def __init__(self, config: Optional[ImageProcessorConfig] = None) -> None:
        self.config = config or ImageProcessorConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size),
        )
        self.trajectory_projector = TrajectoryProjector()

    def enhance_outdoor_lighting(self, bgr_image: np.ndarray) -> np.ndarray:
        """Enhance outdoor contrast and handle harsh sun/shadows using CLAHE in LAB space.

        Applies equalization strictly to the L (luminance) channel to avoid color distortion.
        """
        if not self.config.enable_clahe:
            return bgr_image.copy()

        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self._clahe.apply(l_channel)
        merged_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def extract_structural_edges(self, bgr_image: np.ndarray) -> np.ndarray:
        """Extract structural edge map for obstacle detection and terrain boundary analysis.

        Excludes the lower self-robot vehicle chassis/bumper region.
        """
        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        
        # Self-robot mask: prevent artificial gradient at boundary
        mask_y = h
        if self.config.self_mask_height_ratio > 0.0:
            mask_y = int(h * (1.0 - self.config.self_mask_height_ratio))
            if 0 < mask_y < h:
                gray[mask_y:, :] = gray[mask_y - 1:mask_y, :]

        ksize = self.config.gaussian_blur_ksize
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        edges = cv2.Canny(
            blurred,
            self.config.canny_low_threshold,
            self.config.canny_high_threshold,
        )

        if self.config.self_mask_height_ratio > 0.0:
            edges[mask_y:, :] = 0

        return edges

    def get_ground_roi_polygon(self, height: int, width: int) -> np.ndarray:
        """Compute the ground-plane traversability corridor polygon in camera perspective."""
        y_top = int(height * self.config.ground_roi_top_ratio)
        max_bottom_ratio = 1.0 - self.config.self_mask_height_ratio if self.config.self_mask_height_ratio > 0.0 else 1.0
        effective_bottom_ratio = min(self.config.ground_roi_bottom_ratio, max_bottom_ratio)
        y_bottom = int(height * effective_bottom_ratio)

        top_half_w = int((width * self.config.ground_roi_top_width_ratio) / 2)
        bottom_half_w = int((width * self.config.ground_roi_bottom_width_ratio) / 2)
        cx = width // 2

        pts = np.array([
            [cx - top_half_w, y_top],
            [cx + top_half_w, y_top],
            [cx + bottom_half_w, y_bottom],
            [cx - bottom_half_w, y_bottom],
        ], dtype=np.int32)
        return pts

    def analyze_ground_region(
        self,
        bgr_image: np.ndarray,
        edge_map: np.ndarray,
        ground_pts: np.ndarray,
    ) -> Dict[str, float]:
        """Compute visual statistics within the ground traversability ROI."""
        mask = np.zeros(bgr_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [ground_pts], 255)

        # Ensure self-robot region is excluded from ground ROI
        if self.config.self_mask_height_ratio > 0.0:
            mask_y = int(bgr_image.shape[0] * (1.0 - self.config.self_mask_height_ratio))
            mask[mask_y:, :] = 0

        roi_pixels = cv2.countNonZero(mask)
        if roi_pixels == 0:
            return {'mean_brightness': 0.0, 'edge_density': 0.0}

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(cv2.mean(gray, mask=mask)[0])

        edge_pixels = cv2.countNonZero(cv2.bitwise_and(edge_map, edge_map, mask=mask))
        edge_density = float(edge_pixels) / float(roi_pixels)

        return {
            'mean_brightness': mean_brightness,
            'edge_density': edge_density,
            'roi_area_px': roi_pixels,
        }

    def render_overlay_view(
        self,
        enhanced_bgr: np.ndarray,
        edge_map: np.ndarray,
        ground_pts: np.ndarray,
        metadata: Dict[str, Any],
        ground_stats: Dict[str, float],
    ) -> np.ndarray:
        """Render single-frame augmented perception display with ROI and telemetry HUD."""
        h, w = enhanced_bgr.shape[:2]
        canvas = enhanced_bgr.copy()

        # 1. Overlay translucent ground traversability zone
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [ground_pts], (30, 180, 50))  # Translucent green corridor
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        cv2.polylines(canvas, [ground_pts], isClosed=True, color=(0, 255, 120), thickness=2)

        # Indicate self-robot mask zone if active
        if self.config.self_mask_height_ratio > 0.0:
            mask_y = int(h * (1.0 - self.config.self_mask_height_ratio))
            cv2.rectangle(canvas, (0, mask_y), (w, h), (30, 30, 35), -1)
            cv2.line(canvas, (0, mask_y), (w, mask_y), (80, 80, 100), 1)
            cv2.putText(canvas, "[SELF-ROBOT MASK ACTIVE]", (w // 2 - 80, mask_y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 180), 1, cv2.LINE_AA)

        # Horizon / ground threshold line
        y_top = ground_pts[0][1]
        cv2.line(canvas, (0, y_top), (w, y_top), (0, 200, 255), 1, cv2.LINE_AA)

        # 2. Highlight structural obstacle edges in orange/red
        edge_mask = edge_map > 0
        if np.any(edge_mask):
            canvas[edge_mask] = (0, 80, 255)

        # 3. Draw HUD telemetry banners
        self._render_hud(canvas, h, w, metadata, ground_stats)
        return canvas

    def render_side_by_side_view(
        self,
        enhanced_bgr: np.ndarray,
        edge_map: np.ndarray,
        ground_pts: np.ndarray,
        metadata: Dict[str, Any],
        ground_stats: Dict[str, float],
    ) -> np.ndarray:
        """Render side-by-side diagnostic canvas: Left = Enhanced RGB, Right = Structural Edges."""
        h, w = enhanced_bgr.shape[:2]

        left_canvas = enhanced_bgr.copy()
        overlay = left_canvas.copy()
        cv2.fillPoly(overlay, [ground_pts], (30, 180, 50))
        cv2.addWeighted(overlay, 0.25, left_canvas, 0.75, 0, left_canvas)
        cv2.polylines(left_canvas, [ground_pts], isClosed=True, color=(0, 255, 120), thickness=2)

        # Convert edge map to 3-channel
        edge_bgr = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        cv2.polylines(edge_bgr, [ground_pts], isClosed=True, color=(0, 255, 120), thickness=2)

        # Combine horizontally
        canvas = np.hstack([left_canvas, edge_bgr])

        # Render HUD on left and right headers
        self._render_hud(canvas, h, w * 2, metadata, ground_stats, is_wide=True)
        return canvas

    def _render_hud(
        self,
        canvas: np.ndarray,
        h: int,
        w: int,
        metadata: Dict[str, Any],
        ground_stats: Dict[str, float],
        is_wide: bool = False,
    ) -> None:
        """Render top and bottom telemetry information banners onto canvas."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale_small = 0.38
        scale_title = 0.44
        color_white = (240, 240, 240)
        color_yellow = (0, 230, 255)
        color_green = (80, 255, 80)
        color_cyan = (255, 230, 100)

        # Top banner background (semi-transparent dark bar)
        top_bar_h = 46
        cv2.rectangle(canvas, (0, 0), (w, top_bar_h), (18, 18, 22), -1)
        cv2.line(canvas, (0, top_bar_h), (w, top_bar_h), (70, 70, 80), 1)

        frame_idx = metadata.get('frame_idx', 0)
        frame_id = metadata.get('frame_id', 'camera_link')
        short_frame_id = frame_id.split('/')[-1] if '/' in frame_id else frame_id
        stamp_sec = metadata.get('stamp_sec', 0)
        stamp_nanosec = metadata.get('stamp_nanosec', 0)

        # Top line: Title (left) & Performance (right)
        title_text = f"NAVIGUARD PERCEPTION V1 | Frame #{frame_idx}"
        cv2.putText(canvas, title_text, (8, 18), font, scale_title, color_yellow, 1, cv2.LINE_AA)

        input_fps = metadata.get('input_fps', 0.0)
        proc_latency_ms = metadata.get('proc_latency_ms', 0.0)
        proc_fps = metadata.get('proc_fps', 0.0)
        perf_text = f"In: {input_fps:.1f} FPS | Lat: {proc_latency_ms:.1f}ms | Cap: {proc_fps:.0f} FPS"
        t_sz, _ = cv2.getTextSize(perf_text, font, scale_small, 1)
        cv2.putText(canvas, perf_text, (max(w - t_sz[0] - 8, 300), 18), font, scale_small, color_green, 1, cv2.LINE_AA)

        # Second line: Sim time & Dimensions (left) & Frame ID (right)
        sim_time_str = f"Sim: {stamp_sec}.{stamp_nanosec // 1000000:03d}s | Dim: {metadata.get('orig_w', 640)}x{metadata.get('orig_h', 480)}"
        cv2.putText(canvas, sim_time_str, (8, 37), font, scale_small, color_white, 1, cv2.LINE_AA)

        frame_str = f"Frame: {short_frame_id}"
        f_sz, _ = cv2.getTextSize(frame_str, font, scale_small, 1)
        cv2.putText(canvas, frame_str, (max(w - f_sz[0] - 8, 300), 37), font, scale_small, color_cyan, 1, cv2.LINE_AA)

        # Bottom banner background
        bottom_bar_h = 24
        cv2.rectangle(canvas, (0, h - bottom_bar_h), (w, h), (18, 18, 22), -1)
        cv2.line(canvas, (0, h - bottom_bar_h), (w, h - bottom_bar_h), (70, 70, 80), 1)

        # Bottom Left: Camera Intrinsics
        fx = metadata.get('fx')
        fy = metadata.get('fy')
        cx = metadata.get('cx')
        cy = metadata.get('cy')
        if fx is not None and cx is not None:
            calib_str = f"K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
        else:
            calib_str = "K: Waiting for /camera/camera_info..."
        cv2.putText(canvas, calib_str, (8, h - 7), font, scale_small, (210, 210, 210), 1, cv2.LINE_AA)

        # Bottom Right: Ground ROI indicators
        mean_br = ground_stats.get('mean_brightness', 0.0)
        edge_dens = ground_stats.get('edge_density', 0.0) * 100.0
        ground_text = f"Ground: Brightness={mean_br:.1f} | Edge={edge_dens:.1f}%"
        gt_size, _ = cv2.getTextSize(ground_text, font, scale_small, 1)
        cv2.putText(canvas, ground_text, (max(w - gt_size[0] - 8, 300), h - 7), font, scale_small, (140, 240, 140), 1, cv2.LINE_AA)

    def process_frame(
        self,
        bgr_image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute deterministic perception pipeline on incoming camera image.

        Args:
            bgr_image: Input BGR image numpy array.
            metadata: Context dictionary containing timestamps, frame_id, camera_info, etc.

        Returns:
            Tuple of (processed_bgr_image, diagnostics_dict).
        """
        start_time = time.perf_counter()
        meta = metadata.copy() if metadata else {}

        h, w = bgr_image.shape[:2]
        meta['orig_h'] = h
        meta['orig_w'] = w

        # 1. Outdoor lighting enhancement
        enhanced_bgr = self.enhance_outdoor_lighting(bgr_image)

        # 2. Structural & obstacle edge extraction
        edge_map = self.extract_structural_edges(enhanced_bgr)

        # 3. Ground corridor ROI calculation & analysis
        ground_pts = self.get_ground_roi_polygon(h, w)
        ground_stats = self.analyze_ground_region(enhanced_bgr, edge_map, ground_pts)

        # 4. Render diagnostic visualization based on configured mode
        if self.config.display_mode == 'side_by_side':
            debug_image = self.render_side_by_side_view(
                enhanced_bgr, edge_map, ground_pts, meta, ground_stats
            )
        else:
            debug_image = self.render_overlay_view(
                enhanced_bgr, edge_map, ground_pts, meta, ground_stats
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        diagnostics = {
            'processing_latency_ms': elapsed_ms,
            'ground_stats': ground_stats,
            'image_height': h,
            'image_width': w,
            'enhanced': self.config.enable_clahe,
        }

        return debug_image, diagnostics

    def generate_segmentation_view(
        self,
        bgr_image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate classical terrain traversability and edge segmentation visualization.

        Combines Canny edge detection, perspective ground corridor projection,
        horizon analysis, and chassis masking into a multi-class traversability map.
        Zero fake data: explicitly labelled as classical algorithmic segmentation.
        """
        start_time = time.perf_counter()
        meta = metadata.copy() if metadata else {}
        h, w = bgr_image.shape[:2]

        # 1. Structural edge extraction
        edge_map = self.extract_structural_edges(bgr_image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edge_map, kernel, iterations=1)

        # 2. Geometric horizon & corridor
        y_top = int(h * self.config.ground_roi_top_ratio)
        ground_pts = self.get_ground_roi_polygon(h, w)
        ground_roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(ground_roi_mask, [ground_pts], 255)

        mask_y = h
        if self.config.self_mask_height_ratio > 0.0:
            mask_y = int(h * (1.0 - self.config.self_mask_height_ratio))
            ground_roi_mask[mask_y:, :] = 0
            dilated_edges[mask_y:, :] = 0

        # 3. Construct multi-class color mask
        seg_color = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Class 1: Sky / Far Horizon (top region above ground horizon)
        seg_color[:y_top, :] = (215, 175, 75)   # Sky Blue (BGR)

        # Class 2: Rough Boundary / Off-corridor (below horizon, outside corridor, above chassis)
        seg_color[y_top:mask_y, :] = (20, 160, 210)  # Amber / Earth (BGR)

        # Class 3: Traversable Ground (inside corridor, zero edge detection)
        traversable_mask = (ground_roi_mask > 0) & (dilated_edges == 0)
        seg_color[traversable_mask] = (45, 190, 70)  # Traversable Green (BGR)

        # Class 4: Obstacles / Structural Edges
        obstacle_mask = (dilated_edges > 0)
        obstacle_mask[mask_y:, :] = False
        seg_color[obstacle_mask] = (30, 30, 225)     # Obstacle Red (BGR)

        # Class 5: Self-Robot Chassis Mask
        if mask_y < h:
            seg_color[mask_y:, :] = (35, 35, 40)     # Dark Chassis Gray

        # 4. Alpha blend with original camera image
        canvas = cv2.addWeighted(bgr_image, 0.35, seg_color, 0.65, 0)

        # 5. Real-Time Distance Identification of Objects/Obstacles
        fx = meta.get('fx')
        fy = meta.get('fy')
        cx = meta.get('cx')
        cy = meta.get('cy')
        cmd_vx = float(meta.get('cmd_vx', 0.15))
        cmd_wz = float(meta.get('cmd_wz', 0.0))

        detected_obstacles: List[Dict[str, Any]] = []
        # Find obstacle contours in ground plane region
        obs_roi = np.zeros((h, w), dtype=np.uint8)
        obs_roi[obstacle_mask] = 255
        contours, _ = cv2.findContours(obs_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 30:
                continue
            obx, oby, obw, obh = cv2.boundingRect(cnt)
            # Lowest point represents ground contact
            u_base = obx + obw // 2
            v_base = min(mask_y - 1, oby + obh)

            gx, gy = self.trajectory_projector.project_pixel_to_ground(u_base, v_base, w, h, fx, fy, cx, cy)
            dist_to_base = math.hypot(gx, gy)
            dist_to_bumper = max(0.05, dist_to_base - self.trajectory_projector.FRONT_BUMPER_X_M)
            est_rad = max(0.12, min(0.60, (obw / max(1.0, fx or 381.0)) * gx * 0.5))

            detected_obstacles.append({
                'x_m': gx,
                'y_m': gy,
                'dist_m': dist_to_bumper,
                'radius_m': est_rad,
                'bbox': [obx, oby, obw, obh],
            })

            # Render Real-Time Distance Badge over each object
            tag_color = (30, 30, 240) if dist_to_bumper < 0.8 else ((20, 200, 245) if dist_to_bumper < 1.5 else (50, 240, 90))
            tag_txt = f"{dist_to_bumper:.2f}m"
            cv2.rectangle(canvas, (obx, oby - 16), (obx + 52, oby), (16, 16, 20), -1)
            cv2.rectangle(canvas, (obx, oby - 16), (obx + 52, oby), tag_color, 1)
            cv2.putText(canvas, tag_txt, (obx + 4, oby - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (u_base, v_base), 3, tag_color, -1)

        # Include any external confirmed obstacles passed in metadata
        if 'confirmed_obstacles' in meta and isinstance(meta['confirmed_obstacles'], list):
            for ext_obs in meta['confirmed_obstacles']:
                detected_obstacles.append(ext_obs)

        # 6. Overlay Dynamic Reversing Guidelines (Curved rails, crossbars, PASS/BLOCKED indicator)
        canvas, clearance_info = self.trajectory_projector.render_reversing_overlay(
            canvas, cmd_vx=cmd_vx, cmd_wz=cmd_wz, obstacles=detected_obstacles, metadata=meta
        )

        # 7. Overlay corridor boundary and horizon
        cv2.polylines(canvas, [ground_pts], isClosed=True, color=(100, 255, 100), thickness=2)
        cv2.line(canvas, (0, y_top), (w, y_top), (255, 200, 50), 1, cv2.LINE_AA)

        # 8. Statistics calculation
        corridor_pixels = max(1, cv2.countNonZero(ground_roi_mask))
        trav_pixels = cv2.countNonZero(traversable_mask.astype(np.uint8))
        traversability_pct = float(trav_pixels) / float(corridor_pixels) * 100.0
        edge_pixels_in_corridor = cv2.countNonZero(cv2.bitwise_and(dilated_edges, dilated_edges, mask=ground_roi_mask))
        confidence_score = max(0.10, min(1.0, 1.0 - (float(edge_pixels_in_corridor) / float(corridor_pixels))))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 9. Render Professional HUD & Legend Overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        header_h = 44
        cv2.rectangle(canvas, (0, 0), (w, header_h), (18, 18, 22), -1)
        cv2.line(canvas, (0, header_h), (w, header_h), (60, 60, 70), 1)

        title_str = "SEGMENTATION & DISTANCE IDENTIFICATION"
        cv2.putText(canvas, title_str, (8, 18), font, 0.42, (0, 225, 255), 1, cv2.LINE_AA)

        fps_val = meta.get('input_fps', 0.0)
        min_d_str = f"Min Obs: {clearance_info['min_clearance_m']:.2f}m"
        cv2.putText(canvas, min_d_str, (w - 330, 18), font, 0.36, (255, 200, 50), 1, cv2.LINE_AA)
        perf_str = f"FPS: {fps_val:.1f} | Lat: {elapsed_ms:.1f}ms"
        cv2.putText(canvas, perf_str, (w - 170, 18), font, 0.36, (100, 255, 100), 1, cv2.LINE_AA)

        # Legend Bar
        legend_items = [
            ((45, 190, 70), "Traversable"),
            ((30, 30, 225), "Obstacle+Dist"),
            ((20, 160, 210), "Off-Corridor"),
            ((215, 175, 75), "Far/Sky"),
            ((40, 240, 60), "Guideline"),
        ]
        lx = 8
        for color, name in legend_items:
            cv2.rectangle(canvas, (lx, 26), (lx + 10, 36), color, -1)
            cv2.rectangle(canvas, (lx, 26), (lx + 10, 36), (200, 200, 200), 1)
            cv2.putText(canvas, name, (lx + 14, 35), font, 0.32, (220, 220, 220), 1, cv2.LINE_AA)
            lx += 115

        # Footer banner
        footer_h = 24
        cv2.rectangle(canvas, (0, h - footer_h), (w, h), (18, 18, 22), -1)
        cv2.line(canvas, (0, h - footer_h), (w, h - footer_h), (60, 60, 70), 1)

        status_str = f"Traversability: {traversability_pct:.1f}% | Clearance: {clearance_info['min_clearance_m']:.2f}m | Status: {clearance_info['status']}"
        cv2.putText(canvas, status_str, (8, h - 7), font, 0.35, (180, 255, 180), 1, cv2.LINE_AA)

        seg_diag = {
            'processing_latency_ms': elapsed_ms,
            'traversability_pct': traversability_pct,
            'confidence_score': confidence_score,
            'corridor_pixels': corridor_pixels,
            'obstacle_pixels_in_corridor': edge_pixels_in_corridor,
            'detected_obstacles_count': len(detected_obstacles),
            'min_clearance_m': clearance_info['min_clearance_m'],
            'can_pass': clearance_info['can_pass'],
            'trajectory_status': clearance_info['status'],
        }
        return canvas, seg_diag

    def generate_unified_perception_view(
        self,
        bgr_image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        yolo_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate Unified Multi-Spectral Perception View (Segmentation + YOLO + Perception + Trajectory in One Video).
        
        Merges:
        1. Classical Terrain Traversability Segmentation (trail, obstacle contours, corridor).
        2. YOLOv8 Deep Object Detections with semantic labels and metric distance.
        3. Perception Corridor Guidelines, road boundaries, and horizon line.
        4. Dynamic Automotive Reversing & Path Trajectory with distance crossbars and pass status.
        """
        start_time = time.perf_counter()
        meta = metadata.copy() if metadata else {}
        h, w = bgr_image.shape[:2]

        fx = meta.get('fx')
        fy = meta.get('fy')
        cx = meta.get('cx')
        cy = meta.get('cy')
        cmd_vx = float(meta.get('cmd_vx', 0.15))
        cmd_wz = float(meta.get('cmd_wz', 0.0))

        # 1. Lighting enhancement
        enhanced = self.enhance_outdoor_lighting(bgr_image)

        # 2. Structural Edge Extraction
        edge_map = self.extract_structural_edges(enhanced)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edge_map, kernel, iterations=1)

        # 3. Ground Corridor ROI
        ground_pts = self.get_ground_roi_polygon(h, w)
        ground_roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(ground_roi_mask, [ground_pts], 255)

        mask_y = h
        if self.config.self_mask_height_ratio > 0.0:
            mask_y = int(h * (1.0 - self.config.self_mask_height_ratio))
            ground_roi_mask[mask_y:, :] = 0

        # 4. Color-based Traversability Segmentation
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, v_chan = cv2.split(hsv)

        trail_color_mask = (h_chan >= 12) & (h_chan <= 48) & (s_chan >= 15) & (v_chan >= 40)
        traversable_mask = trail_color_mask & (ground_roi_mask > 0) & (dilated_edges == 0)
        obstacle_mask = (dilated_edges > 0) & (ground_roi_mask > 0)
        off_corridor_mask = (ground_roi_mask > 0) & (~traversable_mask) & (~obstacle_mask)

        # Build semi-transparent multi-spectral segmentation layer
        seg_color = np.zeros_like(bgr_image)
        seg_color[traversable_mask] = (45, 190, 70)       # Vibrant green for traversable trail
        seg_color[obstacle_mask] = (30, 30, 225)          # Bright red for obstacles
        seg_color[off_corridor_mask] = (20, 160, 210)     # Amber for off-corridor
        y_top = ground_pts[0][1]
        seg_color[:y_top, :] = (215, 175, 75)             # Cool slate for sky/horizon

        # Alpha blend with original texture (50% image, 50% segmentation)
        canvas = cv2.addWeighted(bgr_image, 0.50, seg_color, 0.50, 0)

        # 5. Draw Perception Corridor Boundary Lines and Horizon
        cv2.polylines(canvas, [ground_pts], isClosed=True, color=(100, 255, 100), thickness=2)
        cv2.line(canvas, (0, y_top), (w, y_top), (255, 200, 50), 1, cv2.LINE_AA)

        # 6. Real-Time Obstacle Distance Identification (Contours)
        detected_obstacles: List[Dict[str, Any]] = []
        obs_roi = np.zeros((h, w), dtype=np.uint8)
        obs_roi[obstacle_mask] = 255
        contours, _ = cv2.findContours(obs_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 30:
                continue
            obx, oby, obw, obh = cv2.boundingRect(cnt)
            u_base = obx + obw // 2
            v_base = min(mask_y - 1, oby + obh)

            gx, gy = self.trajectory_projector.project_pixel_to_ground(u_base, v_base, w, h, fx, fy, cx, cy)
            dist_to_base = math.hypot(gx, gy)
            dist_to_bumper = max(0.05, dist_to_base - self.trajectory_projector.FRONT_BUMPER_X_M)
            est_rad = max(0.12, min(0.60, (obw / max(1.0, fx or 381.0)) * gx * 0.5))

            detected_obstacles.append({
                'x_m': gx,
                'y_m': gy,
                'dist_m': dist_to_bumper,
                'radius_m': est_rad,
                'bbox': [obx, oby, obw, obh],
            })

            # Real-time distance tag on obstacle contour
            tag_col = (30, 30, 240) if dist_to_bumper < 0.8 else ((20, 200, 245) if dist_to_bumper < 1.5 else (50, 240, 90))
            cv2.rectangle(canvas, (obx, oby - 16), (obx + 52, oby), (16, 16, 20), -1)
            cv2.rectangle(canvas, (obx, oby - 16), (obx + 52, oby), tag_col, 1)
            cv2.putText(canvas, f"{dist_to_bumper:.2f}m", (obx + 4, oby - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (u_base, v_base), 3, tag_col, -1)

        # 7. Merge and Render YOLO Deep Object Detections
        num_yolo = 0
        if yolo_detections:
            for det in yolo_detections:
                bbox = det.get('bbox') or det.get('bounding_box')
                if not bbox and 'bbox_xyxy' in det:
                    x1, y1, x2, y2 = det['bbox_xyxy']
                    bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                if not bbox or len(bbox) < 4:
                    continue
                bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                if bw <= 0 or bh <= 0:
                    continue
                num_yolo += 1
                cname = det.get('class_name', 'obstacle')
                conf = float(det.get('confidence', 0.0))
                traversable = bool(det.get('traversable', False))

                # Distance calculation for YOLO detection
                u_yolo_base = bx + bw // 2
                v_yolo_base = min(mask_y - 1, by + bh)
                gx, gy = self.trajectory_projector.project_pixel_to_ground(u_yolo_base, v_yolo_base, w, h, fx, fy, cx, cy)
                yolo_dist = max(0.05, math.hypot(gx, gy) - self.trajectory_projector.FRONT_BUMPER_X_M)

                detected_obstacles.append({
                    'x_m': gx,
                    'y_m': gy,
                    'dist_m': yolo_dist,
                    'radius_m': max(0.18, min(0.80, (bw / max(1.0, fx or 381.0)) * gx * 0.5)),
                    'bbox': [bx, by, bw, bh],
                })

                # High-visibility YOLO bounding box & badge
                yolo_col = (40, 230, 80) if traversable else ((0, 220, 255) if cname in ('car', 'truck', 'bus') else ((255, 60, 200) if cname == 'person' else (25, 40, 245)))
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), yolo_col, 2)

                # Corner accent brackets
                cr_len = max(6, min(16, bw // 4))
                cv2.line(canvas, (bx, by), (bx + cr_len, by), yolo_col, 3)
                cv2.line(canvas, (bx, by), (bx, by + cr_len), yolo_col, 3)
                cv2.line(canvas, (bx + bw, by), (bx + bw - cr_len, by), yolo_col, 3)
                cv2.line(canvas, (bx + bw, by), (bx + bw, by + cr_len), yolo_col, 3)

                # Detection badge with class, confidence, and metric distance
                badge_str = f"{cname.upper()} {int(conf * 100)}% | {yolo_dist:.2f}m"
                (tw, th), _ = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                cv2.rectangle(canvas, (bx, max(0, by - th - 8)), (bx + tw + 8, by), (14, 16, 22), -1)
                cv2.rectangle(canvas, (bx, max(0, by - th - 8)), (bx + tw + 8, by), yolo_col, 1)
                cv2.putText(canvas, badge_str, (bx + 4, by - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # 8. Dynamic Automotive Reversing Trajectory Overlay
        canvas, clearance_info = self.trajectory_projector.render_reversing_overlay(
            canvas, cmd_vx=cmd_vx, cmd_wz=cmd_wz, obstacles=detected_obstacles, metadata=meta
        )

        # Mask vehicle chassis in dark slate
        if mask_y < h:
            canvas[mask_y:, :] = (28, 30, 36)
            cv2.line(canvas, (0, mask_y), (w, mask_y), (70, 75, 85), 1)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 9. Top Unified HUD Banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        head_h = 44
        cv2.rectangle(canvas, (0, 0), (w, head_h), (14, 16, 22), -1)
        cv2.line(canvas, (0, head_h), (w, head_h), (60, 65, 75), 1)

        cv2.putText(canvas, "UNIFIED PERCEPTION (YOLO + SEGMENTATION + TRAJECTORY)", (8, 18),
                    font, 0.40, (0, 230, 255), 1, cv2.LINE_AA)

        diag_str = f"YOLO: {num_yolo} | Min Clear: {clearance_info['min_clearance_m']:.2f}m | Trajectory: {clearance_info['status']}"
        cv2.putText(canvas, diag_str, (w - 410, 18), font, 0.36, (255, 215, 60), 1, cv2.LINE_AA)

        fps_val = meta.get('input_fps', 0.0)
        cv2.putText(canvas, f"{fps_val:.1f} FPS", (w - 65, 18), font, 0.36, (80, 255, 80), 1, cv2.LINE_AA)

        # Legend Bar
        legend_items = [
            ((45, 190, 70), "Trail/Traversable"),
            ((30, 30, 225), "Obstacle+Dist"),
            ((255, 60, 200), "YOLO Objects"),
            ((255, 200, 50), "Corridor"),
            ((40, 240, 60), "Dynamic Trajectory"),
        ]
        lx = 8
        for color, name in legend_items:
            cv2.rectangle(canvas, (lx, 26), (lx + 10, 36), color, -1)
            cv2.rectangle(canvas, (lx, 26), (lx + 10, 36), (180, 180, 180), 1)
            cv2.putText(canvas, name, (lx + 13, 35), font, 0.30, (220, 220, 220), 1, cv2.LINE_AA)
            lx += 120

        # Footer banner
        footer_h = 24
        cv2.rectangle(canvas, (0, h - footer_h), (w, h), (14, 16, 22), -1)
        cv2.line(canvas, (0, h - footer_h), (w, h - footer_h), (60, 65, 75), 1)
        foot_str = (
            f"All-In-One Vision Active | Detections: {num_yolo} | "
            f"Clearance: {clearance_info['min_clearance_m']:.2f}m | Can Pass: {clearance_info['can_pass']}"
        )
        cv2.putText(canvas, foot_str, (8, h - 7), font, 0.34, (180, 255, 180), 1, cv2.LINE_AA)

        unified_diag = {
            'processing_latency_ms': elapsed_ms,
            'yolo_detections_count': num_yolo,
            'detected_obstacles_count': len(detected_obstacles),
            'min_clearance_m': clearance_info['min_clearance_m'],
            'can_pass': clearance_info['can_pass'],
            'trajectory_status': clearance_info['status'],
        }

        return canvas, unified_diag

    def generate_heatmap_view(
        self,
        bgr_image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        obstacles: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate real-time Distance and Obstacle Proximity Heatmap.

        Computes a continuous metric proximity gradient across the ground plane:
        - Red (0.0 - 0.8m): Critical collision hazard zone near vehicle.
        - Yellow/Amber (0.8 - 1.5m): Caution zone / narrow passage margin.
        - Green (1.5 - 2.5m): Safe traversable corridor.
        - Blue/Purple (> 2.5m): Far-field ground.
        Overlays real-time numerical distance callouts, distance rings, and reversing guidelines.
        """
        start_time = time.perf_counter()
        meta = metadata.copy() if metadata else {}
        h, w = bgr_image.shape[:2]

        fx = meta.get('fx')
        fy = meta.get('fy')
        cx = meta.get('cx')
        cy = meta.get('cy')
        cmd_vx = float(meta.get('cmd_vx', 0.15))
        cmd_wz = float(meta.get('cmd_wz', 0.0))

        y_top = int(h * self.config.ground_roi_top_ratio)
        mask_y = h
        if self.config.self_mask_height_ratio > 0.0:
            mask_y = int(h * (1.0 - self.config.self_mask_height_ratio))

        # Extract edges for obstacle proximity
        edge_map = self.extract_structural_edges(bgr_image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated_edges = cv2.dilate(edge_map, kernel, iterations=2)

        # 1. Generate dense ground distance map
        # Array of row indices v in ground region
        v_indices = np.arange(y_top, mask_y, dtype=np.float32)
        # Compute forward distance x for each row
        if cy is None or fy is None:
            cy_val = h / 2.0
            fy_val = (w / 2.0) / math.tan(math.radians(self.config.ground_roi_top_ratio * 45.0))
        else:
            cy_val = cy
            fy_val = fy

        ray_pitches = np.arctan2(v_indices - cy_val, fy_val) + self.trajectory_projector.camera_pitch_rad
        ray_pitches = np.maximum(0.05, ray_pitches)
        x_dists = self.trajectory_projector.camera_height_m / np.tan(ray_pitches)  # shape (num_rows,)
        # Subtract front bumper
        bumper_dists = np.maximum(0.0, x_dists - self.trajectory_projector.FRONT_BUMPER_X_M)

        # Broadcast into 2D ground distance field
        ground_dist_grid = np.tile(bumper_dists[:, np.newaxis], (1, w))  # (num_rows, w)

        # Normalized distance: close (0m) -> 255 (hot/red), far (4.0m) -> 0 (cool/blue)
        max_map_dist = 3.8
        norm_dist = np.clip(1.0 - (ground_dist_grid / max_map_dist), 0.0, 1.0)
        proximity_gray = (norm_dist * 255.0).astype(np.uint8)

        # Intensify thermal value near obstacle edges
        edge_roi = dilated_edges[y_top:mask_y, :]
        proximity_gray[edge_roi > 0] = np.maximum(proximity_gray[edge_roi > 0], 235)

        # Apply TURBO colormap
        thermal_ground = cv2.applyColorMap(proximity_gray, cv2.COLORMAP_TURBO)

        # Build full-frame canvas
        canvas = bgr_image.copy()
        # Alpha blend 55% thermal heatmap with 45% image texture
        canvas[y_top:mask_y, :] = cv2.addWeighted(
            canvas[y_top:mask_y, :], 0.40, thermal_ground, 0.60, 0
        )

        # Mask vehicle chassis in dark gray
        if mask_y < h:
            canvas[mask_y:, :] = (30, 32, 38)
            cv2.line(canvas, (0, mask_y), (w, mask_y), (80, 85, 95), 1)

        # 2. Draw Iso-Distance Contour Lines across ground (0.5m, 1.0m, 1.5m, 2.0m, 3.0m)
        contour_dists = [0.50, 1.00, 1.50, 2.00, 3.00]
        for c_dist in contour_dists:
            target_x = c_dist + self.trajectory_projector.FRONT_BUMPER_X_M
            # Find closest row v
            row_idx = np.argmin(np.abs(x_dists - target_x))
            v_line = int(y_top + row_idx)
            if y_top <= v_line < mask_y:
                cv2.line(canvas, (20, v_line), (w - 20, v_line), (240, 240, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"D = {c_dist:.1f}m", (w - 75, v_line - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Detect and Label Obstacles with Exact Distance Callouts
        detected_obs: List[Dict[str, Any]] = []
        obs_contours, _ = cv2.findContours(dilated_edges[y_top:mask_y, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_obstacle_dist = 99.0

        for cnt in obs_contours:
            if cv2.contourArea(cnt) < 35:
                continue
            obx, oby_local, obw, obh = cv2.boundingRect(cnt)
            oby = y_top + oby_local
            u_base = obx + obw // 2
            v_base = min(mask_y - 1, oby + obh)

            gx, gy = self.trajectory_projector.project_pixel_to_ground(u_base, v_base, w, h, fx, fy, cx, cy)
            d_bumper = max(0.05, math.hypot(gx, gy) - self.trajectory_projector.FRONT_BUMPER_X_M)
            if d_bumper < min_obstacle_dist:
                min_obstacle_dist = d_bumper

            detected_obs.append({
                'x_m': gx,
                'y_m': gy,
                'dist_m': d_bumper,
                'radius_m': max(0.15, min(0.60, (obw / max(1.0, fx or 381.0)) * gx * 0.5)),
            })

            # High-visibility callout box
            box_col = (30, 30, 240) if d_bumper < 0.8 else ((20, 200, 245) if d_bumper < 1.5 else (50, 240, 90))
            badge_str = f"DIST: {d_bumper:.2f}m"
            badge_w = 88
            cv2.rectangle(canvas, (obx, oby - 18), (obx + badge_w, oby), (12, 14, 20), -1)
            cv2.rectangle(canvas, (obx, oby - 18), (obx + badge_w, oby), box_col, 1)
            cv2.putText(canvas, badge_str, (obx + 4, oby - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (u_base, v_base), 4, box_col, -1)

        # Merge external confirmed obstacles if present
        if obstacles:
            for obs in obstacles:
                detected_obs.append(obs)
                od = float(obs.get('dist_m', math.hypot(obs.get('x_m', 1.0), obs.get('y_m', 0.0))))
                if od < min_obstacle_dist:
                    min_obstacle_dist = od

        min_obs_val = min_obstacle_dist if min_obstacle_dist < 90.0 else 3.5

        # 4. Render Dynamic Automotive Reversing Trajectory Guidelines
        canvas, clearance_info = self.trajectory_projector.render_reversing_overlay(
            canvas, cmd_vx=cmd_vx, cmd_wz=cmd_wz, obstacles=detected_obs, metadata=meta
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Header HUD with Thermal Legend and Telemetry
        font = cv2.FONT_HERSHEY_SIMPLEX
        head_h = 44
        cv2.rectangle(canvas, (0, 0), (w, head_h), (14, 16, 22), -1)
        cv2.line(canvas, (0, head_h), (w, head_h), (60, 65, 75), 1)

        cv2.putText(canvas, "DISTANCE & PROXIMITY HEATMAP", (8, 18), font, 0.42, (0, 225, 255), 1, cv2.LINE_AA)

        obs_hud = f"Nearest: {min_obs_val:.2f}m | Trajectory: {clearance_info['status']}"
        cv2.putText(canvas, obs_hud, (w - 360, 18), font, 0.36, (255, 210, 50), 1, cv2.LINE_AA)

        fps_val = meta.get('input_fps', 0.0)
        cv2.putText(canvas, f"{fps_val:.1f} FPS", (w - 60, 18), font, 0.36, (100, 255, 100), 1, cv2.LINE_AA)

        # Color spectrum bar (Red -> Amber -> Green -> Blue)
        bar_x = 8
        bar_w = 160
        bar_h = 10
        cv2.putText(canvas, "DANGER", (bar_x, 37), font, 0.30, (50, 50, 240), 1, cv2.LINE_AA)
        grad = np.linspace(255, 0, bar_w, dtype=np.uint8)[np.newaxis, :]
        grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_TURBO)
        canvas[28:38, bar_x + 48:bar_x + 48 + bar_w] = grad_color[0, :, :]
        cv2.putText(canvas, "SAFE (>3m)", (bar_x + 54 + bar_w, 37), font, 0.30, (80, 230, 80), 1, cv2.LINE_AA)

        heatmap_diag = {
            'processing_latency_ms': elapsed_ms,
            'nearest_obstacle_dist_m': round(min_obs_val, 2),
            'min_clearance_m': clearance_info['min_clearance_m'],
            'can_pass': clearance_info['can_pass'],
            'trajectory_status': clearance_info['status'],
            'detected_obstacles_count': len(detected_obs),
        }

        return canvas, heatmap_diag
