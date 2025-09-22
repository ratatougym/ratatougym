import numpy as np

def generate_maze_0_arena(sr, **kwargs):
    """Generate a four-room maze arena with connecting tunnels.
    
    Creates a maze with four square rooms connected by tunnels in a cross pattern.
    Each room is positioned at one corner of the maze.
    
    Args:
        sr (float): Spatial resolution of the arena in units per pixel.
        **kwargs: Arena parameters including:
            room_width (float): Width of each room in spatial units. Defaults to 100.
            room_distance (float): Distance between opposite rooms. Defaults to 50.
            tunnel_width (float): Width of connecting tunnels. Defaults to 20.
            vertical (bool): Whether to rotate the maze 90 degrees.
            
    Returns:
        np.ndarray: Arena map where 0 represents free space and 1 represents walls.
    """
    room_width = int(kwargs.get('room_width', 100) / sr) # pixel
    room_distance = int(kwargs.get('room_distance', 50) / sr) # pixel
    tunnel_width = int(kwargs.get('tunnel_width', 20) / sr) # pixel
    width = room_width * 2 + room_distance
    height = room_width * 2 + room_distance
    border = 5

    arena_map = np.ones((height, width))  # Set all to walls
    # Add rooms
    arena_map[0:room_width, 0:room_width] = 0
    arena_map[0:room_width, -room_width:] = 0
    arena_map[-room_width:, 0:room_width] = 0
    arena_map[-room_width:, -room_width:] = 0
    # Add tunnel
    tunnel_start = room_width // 2 - tunnel_width // 2
    # Horizontal tunnels
    arena_map[tunnel_start:(tunnel_start+tunnel_width), room_width:(room_width+room_distance)] = 0
    arena_map[-(tunnel_start+tunnel_width):-(tunnel_start), room_width:(room_width+room_distance)] = 0
    # Vertical tunnels
    arena_map[room_width:(room_width+room_distance), tunnel_start:(tunnel_start+tunnel_width)] = 0
    arena_map[room_width:(room_width+room_distance), -(tunnel_start+tunnel_width):-(tunnel_start)] = 0

    # Add border around the arena
    arena_map = np.pad(arena_map, border, 'constant', constant_values=1)

    if kwargs.get('vertical'):
        arena_map = np.rot90(arena_map, 1)

    return arena_map
