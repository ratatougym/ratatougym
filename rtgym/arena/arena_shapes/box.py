"""Three-dimensional box arena adapted from grid_and_place."""

import numpy as np


def generate_box_arena(sr, **kwargs):
    dimensions = kwargs.get("dimensions", [100, 100, 100])
    assert len(dimensions) == 3, "dimensions must be a list of 3 elements"

    # Convert dimensions from spatial units to pixels (free interior).
    dimensions = np.asarray(dimensions) / sr
    dimensions = dimensions.astype(int)
    if np.any(dimensions < 1):
        raise ValueError("Box dimensions must span at least one grid cell.")

    border = 5
    arena_map = np.zeros(dimensions)
    arena_map = np.pad(arena_map, border, mode="constant", constant_values=1)
    return arena_map
