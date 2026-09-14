import numpy as np
from naviguard_rellis.label_loader import (
    get_class_name,
    get_class_color,
    get_traversability,
    colorize_label_mask,
    RELLIS_CLASSES
)


def test_class_names():
    assert get_class_name(1) == 'dirt'
    assert get_class_name(3) == 'grass'
    assert get_class_name(4) == 'tree'
    assert get_class_name(15) == 'log'
    assert get_class_name(33) == 'mud'
    assert 'unknown' in get_class_name(999)


def test_traversability():
    assert get_traversability(1) == 'traversable'
    assert get_traversability(3) == 'traversable'
    assert get_traversability(4) == 'obstacle'
    assert get_traversability(15) == 'obstacle'
    assert get_traversability(33) == 'slip_hazard'


def test_colorize_label_mask():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, :] = 1   # dirt
    mask[5:10, :] = 4  # tree
    rgb = colorize_label_mask(mask)
    assert rgb.shape == (10, 10, 3)
    # Tree is (0, 255, 0)
    assert rgb[6, 0, 1] == 255
