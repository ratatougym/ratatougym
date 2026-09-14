"""Hairpin arena and waypoints adapted from grid_and_place."""

import numpy as np


def generate_hairpin_arena(sr, **kwargs):
    # Convert alley and turn dimensions to grid cells.
    n_alleys = kwargs.get("n_alleys", 10)
    n_alleys = int(n_alleys)
    alley_width = kwargs.get("alley_width", 15) / sr
    alley_width = int(alley_width)
    alley_height = kwargs.get("alley_height", 100) / sr
    alley_height = int(alley_height)

    # Set wall thickness and the gap used for each turn.
    wall_thickness = kwargs.get("wall_thickness", 1) / sr
    wall_thickness = int(wall_thickness)
    default_gap = kwargs.get("alley_width", 15)
    turn_gap = kwargs.get("turn_gap", default_gap) / sr
    turn_gap = int(turn_gap)
    border = 5

    assert n_alleys >= 2, f"n_alleys must be >= 2, got {n_alleys}"
    assert turn_gap < alley_height, (
        f"turn_gap ({turn_gap}) must be smaller than alley_height ({alley_height})"
    )

    # Allocate the alley interiors before inserting alternating walls.
    height = alley_height
    width = n_alleys * alley_width + (n_alleys - 1) * wall_thickness
    arena_map = np.zeros((height, width), dtype=np.float32)

    # Walls between alleys.
    for i in range(n_alleys - 1):
        col_start = (i + 1) * alley_width + i * wall_thickness
        col_end = col_start + wall_thickness
        if i % 2 == 0:
            arena_map[turn_gap:, col_start:col_end] = 1  # gap at top
        else:
            arena_map[: height - turn_gap, col_start:col_end] = 1  # gap at bottom

    # Border (walls on all four sides)
    arena_map = np.pad(arena_map, border, mode="constant", constant_values=1)

    if kwargs.get("vertical"):
        arena_map = np.rot90(arena_map, 1, (0, 1))

    return arena_map


def hairpin_waypoints(sr, n_alleys=10, alley_width=15, alley_height=100,
                      wall_thickness=2, turn_gap=None, border=5, direction='forward'):
    """Return grid-coordinate waypoints for the unrotated hairpin layout."""
    alley_width_px = int(alley_width / sr)
    alley_height_px = int(alley_height / sr)
    wall_thickness_px = int(wall_thickness / sr)
    if turn_gap is None:
        turn_gap_px = alley_width_px
    else:
        turn_gap_px = int(turn_gap / sr)

    col_centers = [
        border + i * alley_width_px + i * wall_thickness_px + alley_width_px // 2
        for i in range(n_alleys)
    ]
    # Rows inside the alleys — stay a few pixels inside the turn gap so the
    # waypoint is reachable without clipping the wall.
    margin = max(2, turn_gap_px // 3)
    top_row = border + margin
    bottom_row = border + alley_height_px - 1 - margin

    waypoints = [(bottom_row, col_centers[0])]
    for i in range(n_alleys):
        if i % 2 == 0:
            # Going up alley i
            waypoints.append((top_row, col_centers[i]))
            if i + 1 < n_alleys:
                waypoints.append((top_row, col_centers[i + 1]))
        else:
            # Going down alley i
            waypoints.append((bottom_row, col_centers[i]))
            if i + 1 < n_alleys:
                waypoints.append((bottom_row, col_centers[i + 1]))

    if direction == "backward":
        waypoints = waypoints[::-1]
    elif direction != "forward":
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction}")

    return waypoints
