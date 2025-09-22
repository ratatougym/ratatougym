import numpy as np


def point_in_triangle(p, p0, p1, p2):
    # Barycentric coordinates
    A = 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])
    sign = -1 if A < 0 else 1
    s = (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1]) * sign
    t = (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1]) * sign

    return s > 0 and t > 0 and (s + t) < 2 * A * sign


def generate_triangle_arena(sr, **kwargs):
    # arena_map size
    width = int(kwargs.get('length', 100) / sr) # pixel
    height = int(kwargs.get('length', 100) / sr / 2 * np.sqrt(3)) # pixel

    border = 5

    # Triangle vertices
    p0 = np.array([0, 0])
    p1 = np.array([width, 0])
    p2 = np.array([width/2, height])

    # Create empty arena_map
    arena_map = np.ones((width, height)) # Fill everything as wall

    # Fill in the triangle
    for x in range(width):
        for y in range(height):
            if point_in_triangle(np.array([x, y]), p0, p1, p2):
                arena_map[x, y] = 0 # Cut out the triangle

    arena_map = np.rot90(arena_map, 1)
    arena_map = np.pad(arena_map, border, 'constant', constant_values=1) # Add additional border

    return arena_map