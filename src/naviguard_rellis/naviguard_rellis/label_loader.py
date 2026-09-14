"""
Label Loader for RELLIS-3D Dataset.

Defines the official 20-class semantic taxonomy, RGB color palette, and traversability mapping.
Used strictly for offline evaluation and benchmark analysis.
"""

from typing import Dict, Tuple, Optional
import numpy as np

# RELLIS-3D 20-class taxonomy (class_id -> class_name)
RELLIS_CLASSES: Dict[int, str] = {
    0: 'void',
    1: 'dirt',
    3: 'grass',
    4: 'tree',
    5: 'pole',
    6: 'water',
    7: 'sky',
    8: 'vehicle',
    9: 'object',
    10: 'asphalt',
    12: 'building',
    15: 'log',
    17: 'person',
    18: 'fence',
    19: 'bush',
    23: 'concrete',
    27: 'barrier',
    31: 'puddle',
    33: 'mud',
    34: 'rubble'
}

# Color palette for visualization (RGB)
RELLIS_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),          # void: black
    1: (108, 64, 20),      # dirt: brown
    3: (0, 102, 0),        # grass: green
    4: (0, 255, 0),        # tree: bright green
    5: (153, 153, 153),    # pole: gray
    6: (0, 0, 255),        # water: blue
    7: (135, 206, 235),    # sky: sky blue
    8: (255, 0, 0),        # vehicle: red
    9: (255, 128, 0),      # object: orange
    10: (50, 50, 50),      # asphalt: dark gray
    12: (128, 0, 128),     # building: purple
    15: (139, 69, 19),     # log: saddle brown
    17: (255, 20, 147),    # person: deep pink
    18: (205, 133, 63),    # fence: peru
    19: (34, 139, 34),     # bush: forest green
    23: (192, 192, 192),   # concrete: silver
    27: (255, 69, 0),      # barrier: red-orange
    31: (70, 130, 180),    # puddle: steel blue
    33: (85, 45, 15),      # mud: dark mud brown
    34: (160, 82, 45),     # rubble: sienna
}

# Traversability categories for off-road analysis
TRAVERSABILITY_MAP: Dict[int, str] = {
    0: 'unknown',
    1: 'traversable',      # dirt
    3: 'traversable',      # grass
    4: 'obstacle',         # tree
    5: 'obstacle',         # pole
    6: 'lethal_hazard',    # deep water
    7: 'sky',
    8: 'obstacle',         # vehicle
    9: 'obstacle',         # object
    10: 'traversable',     # asphalt
    12: 'obstacle',        # building
    15: 'obstacle',        # fallen log
    17: 'obstacle',        # person
    18: 'obstacle',        # fence
    19: 'rough_hazard',    # dense bush
    23: 'traversable',     # concrete
    27: 'obstacle',        # barrier
    31: 'slip_hazard',     # puddle
    33: 'slip_hazard',     # mud
    34: 'rough_hazard',    # rubble
}


def get_class_name(class_id: int) -> str:
    """Return the name for a class ID or 'unknown'."""
    return RELLIS_CLASSES.get(class_id, f'unknown_{class_id}')


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Return RGB tuple for class ID."""
    return RELLIS_COLOR_MAP.get(class_id, (128, 128, 128))


def get_traversability(class_id: int) -> str:
    """Return traversability category string."""
    return TRAVERSABILITY_MAP.get(class_id, 'unknown')


def colorize_label_mask(label_mask: np.ndarray) -> np.ndarray:
    """Convert 2D uint8/uint16 label mask to 3D RGB uint8 image."""
    h, w = label_mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in RELLIS_COLOR_MAP.items():
        mask = (label_mask == class_id)
        if np.any(mask):
            rgb[mask] = color
    return rgb
