"""RELLIS-3D Outdoor Terrain and Heightmap Builder for NAVIGUARD Simulation.

Processes 3D LiDAR point clouds or topographical coordinates into
digital elevation models (DEM), heightmaps, and traversability friction zones.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class TerrainSurfaceType:
    TRAIL = 0       # High traction dirt/gravel path (mu = 0.85)
    GRASS = 1       # Standard off-road vegetation (mu = 0.60)
    MUD = 2         # Low friction slip zone (mu = 0.22)
    OBSTACLE = 3    # Lethal non-traversable rock/log (mu = 0.70)


class TerrainBuilder:
    """Constructs elevation grids and friction maps from outdoor 3D spatial points."""

    def __init__(
        self,
        resolution_m: float = 0.20,
        bounds_m: Tuple[float, float, float, float] = (-30.0, 30.0, -30.0, 30.0),
    ) -> None:
        self.resolution = resolution_m
        self.min_x, self.max_x, self.min_y, self.max_y = bounds_m
        self.width_cells = int(np.ceil((self.max_x - self.min_x) / self.resolution))
        self.height_cells = int(np.ceil((self.max_y - self.min_y) / self.resolution))

    def build_elevation_grid(
        self,
        points: np.ndarray,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Bin 3D point cloud (N, 3) into a 2D regular digital elevation model (DEM).

        Returns 2D numpy array [height_cells, width_cells] with elevation z in meters.
        """
        grid = np.full((self.height_cells, self.width_cells), fill_value, dtype=np.float32)
        count = np.zeros((self.height_cells, self.width_cells), dtype=np.int32)

        if points is None or len(points) == 0:
            return grid

        # Filter points within spatial bounding box
        valid_mask = (
            (points[:, 0] >= self.min_x) & (points[:, 0] < self.max_x) &
            (points[:, 1] >= self.min_y) & (points[:, 1] < self.max_y)
        )
        pts = points[valid_mask]
        if len(pts) == 0:
            return grid

        gx = np.floor((pts[:, 0] - self.min_x) / self.resolution).astype(int)
        gy = np.floor((pts[:, 1] - self.min_y) / self.resolution).astype(int)

        np.add.at(grid, (gy, gx), pts[:, 2])
        np.add.at(count, (gy, gx), 1)

        nonzero_mask = count > 0
        grid[nonzero_mask] /= count[nonzero_mask]
        grid[~nonzero_mask] = fill_value
        return grid

    def classify_surfaces(
        self,
        elevation_grid: np.ndarray,
        trail_corridor_width_m: float = 3.5,
        mud_patches: Optional[List[Tuple[float, float, float]]] = None,
        obstacle_slope_threshold: float = 0.25,
    ) -> np.ndarray:
        """Classify each grid cell into a terrain surface type."""
        surface_map = np.full(elevation_grid.shape, TerrainSurfaceType.GRASS, dtype=np.int8)

        # 1. Identify primary dirt trail corridor along central axis
        half_w_cells = int((trail_corridor_width_m / 2.0) / self.resolution)
        cy = self.height_cells // 2
        y_min = max(0, cy - half_w_cells)
        y_max = min(self.height_cells, cy + half_w_cells)
        surface_map[y_min:y_max, :] = TerrainSurfaceType.TRAIL

        # 2. Check terrain slope / roughness for obstacles
        gy, gx = np.gradient(elevation_grid, self.resolution)
        slope = np.hypot(gx, gy)
        surface_map[slope >= obstacle_slope_threshold] = TerrainSurfaceType.OBSTACLE

        # 3. Apply low-traction mud patches: (center_x, center_y, radius_m)
        if mud_patches:
            for mx, my, mr in mud_patches:
                rad_cells = int(np.ceil(mr / self.resolution))
                cx_cell = int((mx - self.min_x) / self.resolution)
                cy_cell = int((my - self.min_y) / self.resolution)
                r_min_y = max(0, cy_cell - rad_cells)
                r_max_y = min(self.height_cells, cy_cell + rad_cells + 1)
                r_min_x = max(0, cx_cell - rad_cells)
                r_max_x = min(self.width_cells, cx_cell + rad_cells + 1)
                for y in range(r_min_y, r_max_y):
                    for x in range(r_min_x, r_max_x):
                        d = math.hypot((x - cx_cell) * self.resolution, (y - cy_cell) * self.resolution)
                        if d <= mr and surface_map[y, x] != TerrainSurfaceType.OBSTACLE:
                            surface_map[y, x] = TerrainSurfaceType.MUD

        return surface_map

    def get_friction_coefficients(self, surface_type: int) -> Tuple[float, float]:
        """Return (mu, mu2) ODE friction coefficients for a surface type."""
        if surface_type == TerrainSurfaceType.TRAIL:
            return 0.85, 0.80
        elif surface_type == TerrainSurfaceType.MUD:
            return 0.22, 0.20
        elif surface_type == TerrainSurfaceType.OBSTACLE:
            return 0.70, 0.70
        else:  # GRASS
            return 0.60, 0.55
