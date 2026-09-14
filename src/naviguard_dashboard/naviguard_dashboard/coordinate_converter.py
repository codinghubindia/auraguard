"""
Map and Canvas Coordinate Converter for NAVIGUARD Operator Dashboard.

Provides bidirectional coordinate transformations:
1. World coordinates (x, y in meters, map frame)
2. OccupancyGrid cell coordinates (col, row)
3. UI Canvas / Image coordinates (pixel_u, pixel_v)

Handles ROS REP-103 conventions (X forward/east, Y left/north)
and browser HTML5 canvas conventions (X right, Y down).
"""

from typing import Tuple, Optional


class MapCoordinateConverter:
    """Transforms coordinates between ROS Map space and UI Canvas space."""

    def __init__(
        self,
        resolution: float = 0.05,
        width_cells: int = 600,
        height_cells: int = 600,
        origin_x: float = -15.0,
        origin_y: float = -15.0,
        canvas_width: int = 600,
        canvas_height: int = 600,
    ):

        self.resolution = max(1e-4, float(resolution))
        self.width_cells = max(1, int(width_cells))
        self.height_cells = max(1, int(height_cells))
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.canvas_width = max(1, int(canvas_width))
        self.canvas_height = max(1, int(canvas_height))

    def update_map_meta(
        self,
        resolution: float,
        width: int,
        height: int,
        origin_x: float,
        origin_y: float,
    ) -> None:
        """Update map dimensions and origin when a new OccupancyGrid arrives."""
        self.resolution = max(1e-4, float(resolution))
        self.width_cells = max(1, int(width))
        self.height_cells = max(1, int(height))
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid cell (col, row)."""
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid cell (col, row) to world coordinates (center of cell in meters)."""
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def grid_to_canvas(self, col: int, row: int) -> Tuple[float, float]:
        """Convert grid cell (col, row) to UI canvas pixel (u, v)."""
        scale_x = self.canvas_width / self.width_cells
        scale_y = self.canvas_height / self.height_cells
        u = col * scale_x
        # Invert Y for canvas coordinate system (row 0 is bottom in ROS, top in canvas)
        v = (self.height_cells - 1 - row) * scale_y
        return u, v

    def canvas_to_grid(self, u: float, v: float) -> Tuple[int, int]:
        """Convert UI canvas pixel (u, v) to grid cell (col, row)."""
        scale_x = self.width_cells / self.canvas_width
        scale_y = self.height_cells / self.canvas_height
        col = int(u * scale_x)
        row = int((self.canvas_height - 1 - v) * scale_y)
        return col, row

    def world_to_canvas(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates (meters) directly to canvas pixel (u, v)."""
        col = (x - self.origin_x) / self.resolution
        row = (y - self.origin_y) / self.resolution
        scale_x = self.canvas_width / self.width_cells
        scale_y = self.canvas_height / self.height_cells
        u = col * scale_x
        v = (self.height_cells - 1 - row) * scale_y
        return u, v

    def canvas_to_world(self, u: float, v: float) -> Tuple[float, float]:
        """Convert canvas pixel (u, v) directly to world coordinates (meters)."""
        col, row = self.canvas_to_grid(u, v)
        return self.grid_to_world(col, row)

    def is_in_grid_bounds(self, col: int, row: int) -> bool:
        """Check if grid cell lies within map matrix dimensions."""
        return 0 <= col < self.width_cells and 0 <= row < self.height_cells

    def is_in_world_bounds(self, x: float, y: float) -> bool:
        """Check if world coordinates fall within map boundaries."""
        col, row = self.world_to_grid(x, y)
        return self.is_in_grid_bounds(col, row)
