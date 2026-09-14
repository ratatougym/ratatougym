"""Carpenter-room layout adapted from grid_and_place."""

import numpy as np


def generate_carpenter_rooms_arena(sr, **kwargs):
    # Convert room and opening sizes to grid cells.
    room_width = kwargs.get("room_width", 90) / sr
    room_width = int(room_width)
    room_height = kwargs.get("room_height", 90) / sr
    room_height = int(room_height)

    # Set corridor, wall and doorway dimensions.
    corridor_width = kwargs.get("corridor_width", 40) / sr
    corridor_width = int(corridor_width)
    wall_thickness = kwargs.get("wall_thickness", 1) / sr
    wall_thickness = int(wall_thickness)
    opening_width = kwargs.get("opening_width", 20) / sr
    opening_width = int(opening_width)
    border = 5

    assert room_width > 0 and room_height > 0, "rooms must have positive dims"
    assert corridor_width > 0, "corridor_width must be positive"
    assert opening_width > 0, "opening_width must be positive"
    assert opening_width < room_width, (
        f"opening_width ({opening_width}) must be < room_width ({room_width})"
    )

    # Allocate both rooms and the shared corridor.
    total_width = 2 * room_width + wall_thickness
    total_height = corridor_width + wall_thickness + room_height
    arena_map = np.zeros((total_height, total_width), dtype=np.float32)

    # Row ranges
    wall_r0 = corridor_width
    wall_r1 = corridor_width + wall_thickness
    room_r0 = wall_r1
    room_r1 = room_r0 + room_height

    # North wall between the corridor and the two rooms, full width.
    arena_map[wall_r0:wall_r1, :] = 1

    # Solid dividing wall between rooms A and B, full room height.
    div_c0 = room_width
    div_c1 = div_c0 + wall_thickness
    arena_map[room_r0:room_r1, div_c0:div_c1] = 1

    # Opening in each room's north wall, centered on the sub-room. 
    half = opening_width // 2
    b_open_center = room_width // 2  # center of room B
    a_open_center = div_c1 + room_width // 2  # center of room A

    b_c0 = max(0, b_open_center - half)
    b_c1 = min(div_c0, b_open_center + (opening_width - half))
    a_c0 = max(div_c1, a_open_center - half)
    a_c1 = min(total_width, a_open_center + (opening_width - half))

    arena_map[wall_r0:wall_r1, b_c0:b_c1] = 0
    arena_map[wall_r0:wall_r1, a_c0:a_c1] = 0

    # Enclose with a wall border.
    arena_map = np.pad(arena_map, border, mode="constant", constant_values=1)

    if kwargs.get("vertical"):
        arena_map = np.rot90(arena_map, 1, (0, 1))

    return arena_map
