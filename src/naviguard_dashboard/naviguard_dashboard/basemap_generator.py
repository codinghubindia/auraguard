"""
Orthographic RELLIS Outdoor Basemap Generator for NAVIGUARD Operator Dashboard.

Renders an accurate top-down 2D basemap of rellis_outdoor_world.sdf,
matching world coordinates [-15, +15]m at 0.05m resolution (600x600 px).
"""

import math
from typing import Tuple, List
import cv2
import numpy as np


def world_to_px(
    x: float,
    y: float,
    origin_x: float = -15.0,
    origin_y: float = -15.0,
    resolution: float = 0.05,
    height: int = 600,
) -> Tuple[int, int]:
    """Convert world coordinates (meters) to image pixel (u, v)."""
    u = int((x - origin_x) / resolution)
    v = int((height - 1) - (y - origin_y) / resolution)
    return u, v


def get_rotated_box_corners(
    cx: float,
    cy: float,
    length: float,
    width: float,
    yaw: float,
    origin_x: float = -15.0,
    origin_y: float = -15.0,
    resolution: float = 0.05,
    height: int = 600,
) -> np.ndarray:
    """Compute polygon points for a rotated rectangular box."""
    hl = length / 2.0
    hw = width / 2.0
    cos_t = math.cos(yaw)
    sin_t = math.sin(yaw)

    local_corners = [
        (-hl, -hw),
        (hl, -hw),
        (hl, hw),
        (-hl, hw),
    ]

    pts = []
    for dx, dy in local_corners:
        wx = cx + dx * cos_t - dy * sin_t
        wy = cy + dx * sin_t + dy * cos_t
        u, v = world_to_px(wx, wy, origin_x, origin_y, resolution, height)
        pts.append([u, v])

    return np.array(pts, dtype=np.int32)


def generate_rellis_basemap(
    width: int = 600,
    height: int = 600,
    resolution: float = 0.05,
    origin_x: float = -15.0,
    origin_y: float = -15.0,
) -> np.ndarray:
    """Generate high-fidelity orthographic rendering of the RELLIS outdoor world."""
    np.random.seed(12345)
    # Natural outdoor grass background with subtle procedural variation
    base_color = np.array([52, 92, 58], dtype=np.float32)  # Lush outdoor green (BGR)
    noise = np.random.normal(0, 4, (height, width, 3)).astype(np.float32)
    img = np.clip(base_color + noise, 0, 255).astype(np.uint8)

    # 5-meter grid lines
    grid_color = (42, 75, 48)
    for gx in range(int(origin_x), int(origin_x + width * resolution) + 1, 5):
        u, _ = world_to_px(float(gx), 0.0, origin_x, origin_y, resolution, height)
        if 0 <= u < width:
            cv2.line(img, (u, 0), (u, height), grid_color, 1)
            cv2.putText(img, f"{gx:+d}m", (u + 3, height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (90, 130, 95), 1, cv2.LINE_AA)

    for gy in range(int(origin_y), int(origin_y + height * resolution) + 1, 5):
        _, v = world_to_px(0.0, float(gy), origin_x, origin_y, resolution, height)
        if 0 <= v < height:
            cv2.line(img, (0, v), (width, v), grid_color, 1)
            cv2.putText(img, f"{gy:+d}m", (6, v - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (90, 130, 95), 1, cv2.LINE_AA)

    # World Boundary Fence Line (+/- 14.5m)
    b_min_u, b_max_v = world_to_px(-14.5, -14.5, origin_x, origin_y, resolution, height)
    b_max_u, b_min_v = world_to_px(14.5, 14.5, origin_x, origin_y, resolution, height)
    cv2.rectangle(img, (b_min_u, b_min_v), (b_max_u, b_max_v), (35, 60, 40), 1, cv2.LINE_AA)

    # Main Dirt Trail Segment 1: center (5.0, 0.0), size 12.0m x 3.2m, yaw 0
    trail_color = (72, 112, 158)       # Warm earth / dirt brown (BGR)
    trail_border = (55, 88, 126)
    corners_t1 = get_rotated_box_corners(5.0, 0.0, 12.0, 3.2, 0.0, origin_x, origin_y, resolution, height)
    cv2.fillPoly(img, [corners_t1], trail_color)
    cv2.polylines(img, [corners_t1], True, trail_border, 2)

    # Main Dirt Trail Segment 2: center (14.0, 3.5), size 10.0m x 3.2m, yaw 0.52 rad
    corners_t2 = get_rotated_box_corners(14.0, 3.5, 10.0, 3.2, 0.52, origin_x, origin_y, resolution, height)
    cv2.fillPoly(img, [corners_t2], trail_color)
    cv2.polylines(img, [corners_t2], True, trail_border, 2)

    # Trail Wheel Ruts / Track Lines
    rut_color = (60, 95, 135)
    r1_u1, r1_v1 = world_to_px(-0.8, 0.7, origin_x, origin_y, resolution, height)
    r1_u2, r1_v2 = world_to_px(10.8, 0.7, origin_x, origin_y, resolution, height)
    cv2.line(img, (r1_u1, r1_v1), (r1_u2, r1_v2), rut_color, 1, cv2.LINE_AA)

    r2_u1, r2_v1 = world_to_px(-0.8, -0.7, origin_x, origin_y, resolution, height)
    r2_u2, r2_v2 = world_to_px(10.8, -0.7, origin_x, origin_y, resolution, height)
    cv2.line(img, (r2_u1, r2_v1), (r2_u2, r2_v2), rut_color, 1, cv2.LINE_AA)

    # Mud Slip Patch: center (6.0, 0.5), size 2.2m x 1.8m, yaw 0.2 rad
    mud_color = (38, 56, 78)           # Dark sticky mud (BGR)
    corners_mud = get_rotated_box_corners(6.0, 0.5, 2.2, 1.8, 0.2, origin_x, origin_y, resolution, height)
    cv2.fillPoly(img, [corners_mud], mud_color)
    cv2.polylines(img, [corners_mud], True, (28, 42, 60), 1)
    mu_u, mu_v = world_to_px(6.0, 0.5, origin_x, origin_y, resolution, height)
    cv2.putText(img, "MUD", (mu_u - 14, mu_v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 190, 205), 1, cv2.LINE_AA)

    # Fallen Log Obstacle: center (8.0, -0.8), length 2.8m, diameter 0.36m, yaw 0.4 rad
    log_color = (28, 46, 72)           # Timber dark brown
    corners_log = get_rotated_box_corners(8.0, -0.8, 2.8, 0.36, 0.4, origin_x, origin_y, resolution, height)
    cv2.fillPoly(img, [corners_log], log_color)
    cv2.polylines(img, [corners_log], True, (18, 30, 48), 2)
    lu, lv = world_to_px(8.0, -0.8, origin_x, origin_y, resolution, height)
    cv2.putText(img, "LOG", (lu - 12, lv - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (230, 180, 100), 1, cv2.LINE_AA)

    # Rock Cairn 1: center (2.0, -2.4), size 0.9m x 0.8m, yaw 0.45 rad
    rock_color = (140, 140, 145)
    corners_rock = get_rotated_box_corners(2.0, -2.4, 0.9, 0.8, 0.45, origin_x, origin_y, resolution, height)
    cv2.fillPoly(img, [corners_rock], rock_color)
    cv2.polylines(img, [corners_rock], True, (95, 95, 100), 1)
    rk_u, rk_v = world_to_px(2.0, -2.4, origin_x, origin_y, resolution, height)
    cv2.putText(img, "ROCK", (rk_u - 15, rk_v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (210, 210, 215), 1, cv2.LINE_AA)

    # Rock Cluster 2: center (11.0, -2.2), size 1.1m x 0.9m, yaw 0.8 rad
    corners_rock2 = get_rotated_box_corners(11.0, -2.2, 1.1, 0.9, 0.8, origin_x, origin_y, resolution, height)
    cv2.fillPoly(img, [corners_rock2], rock_color)
    cv2.polylines(img, [corners_rock2], True, (95, 95, 100), 1)

    # Side Boulder (Impassable gap on shoulder): center (8.5, 1.35), radius 0.18m
    sb_u, sb_v = world_to_px(8.5, 1.35, origin_x, origin_y, resolution, height)
    cv2.circle(img, (sb_u, sb_v), int(0.18 / resolution), rock_color, -1)
    cv2.circle(img, (sb_u, sb_v), int(0.18 / resolution), (95, 95, 100), 1)

    # Trail Border Post Markers
    post_u1, post_v1 = world_to_px(6.5, 1.6, origin_x, origin_y, resolution, height)
    cv2.circle(img, (post_u1, post_v1), int(0.10 / resolution), (45, 140, 160), -1)
    post_u2, post_v2 = world_to_px(10.5, -1.5, origin_x, origin_y, resolution, height)
    cv2.circle(img, (post_u2, post_v2), int(0.10 / resolution), (45, 140, 160), -1)

    # Trees (Foliage canopy + trunk center)
    trees = [
        (3.5, 2.6, 1.5, 0.28, "Live Oak"),
        (4.2, -2.5, 1.1, 0.26, "Cedar"),
        # Twin Gateway Trees forming dedicated 0.66m narrow passage at x=8.5m:
        (8.5, 0.57, 1.3, 0.24, "Gateway L"),
        (8.5, -0.57, 1.3, 0.24, "Gateway R"),
        (12.2, 2.8, 1.6, 0.30, "Live Oak"),
        (16.0, 5.2, 1.2, 0.25, "Cedar"),
    ]
    foliage_color = (36, 122, 50)
    foliage_border = (26, 92, 38)
    trunk_color = (28, 45, 75)

    for tx, ty, f_rad, t_rad, label in trees:
        tu, tv = world_to_px(tx, ty, origin_x, origin_y, resolution, height)
        f_px = int(f_rad / resolution)
        t_px = max(2, int(t_rad / resolution))
        cv2.circle(img, (tu, tv), f_px, foliage_color, -1)
        cv2.circle(img, (tu, tv), f_px, foliage_border, 2)
        cv2.circle(img, (tu, tv), t_px, trunk_color, -1)
        cv2.putText(img, "TREE", (tu - 14, tv - f_px - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (160, 210, 160), 1, cv2.LINE_AA)

    # Narrow Gateway Passage Annotation
    gw_u, gw_v = world_to_px(8.5, 0.0, origin_x, origin_y, resolution, height)
    cv2.putText(img, "PASSAGE 0.66m", (gw_u - 35, gw_v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (5, 241, 135), 1, cv2.LINE_AA)

    # Start marker (X=0, Y=0)
    start_u, start_v = world_to_px(0.0, 0.0, origin_x, origin_y, resolution, height)
    cv2.drawMarker(img, (start_u, start_v), (240, 240, 240), cv2.MARKER_CROSS, 16, 1)
    cv2.circle(img, (start_u, start_v), 8, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, "START (0,0)", (start_u + 10, start_v - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)

    # Trail Guide Text
    tr_u, tr_v = world_to_px(2.5, 0.0, origin_x, origin_y, resolution, height)
    cv2.putText(img, "MAIN TRAIL", (tr_u - 24, tr_v + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 215, 230), 1, cv2.LINE_AA)

    # Coordinate annotations / Compass
    cv2.arrowedLine(img, (40, 55), (40, 25), (220, 240, 220), 2, tipLength=0.3)
    cv2.putText(img, "N (+Y)", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 240, 220), 1, cv2.LINE_AA)
    cv2.arrowedLine(img, (40, 55), (70, 55), (220, 240, 220), 2, tipLength=0.3)
    cv2.putText(img, "E (+X)", (75, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 240, 220), 1, cv2.LINE_AA)

    # Scale Bar (5 meters)
    s_x1, s_y = 40, height - 30
    s_x2 = int(s_x1 + 5.0 / resolution)
    cv2.line(img, (s_x1, s_y), (s_x2, s_y), (220, 230, 220), 2)
    cv2.line(img, (s_x1, s_y - 4), (s_x1, s_y + 4), (220, 230, 220), 2)
    cv2.line(img, (s_x2, s_y - 4), (s_x2, s_y + 4), (220, 230, 220), 2)
    cv2.putText(img, "5.0 m", (int((s_x1 + s_x2) / 2) - 15, s_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220, 230, 220), 1, cv2.LINE_AA)

    # Truthful RELLIS Attribution Banner
    cv2.putText(img, "RELLIS-INSPIRED SYNTHETIC ENVIRONMENT (30m x 30m, 0.05m/px)", (width - 340, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 210, 180), 1, cv2.LINE_AA)

    return img
