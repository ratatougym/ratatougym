import numpy as np

def generate_two_rooms_arena(sr, **kwargs):
    """Generate a two-room arena connected by a tunnel.
    
    Creates two rectangular rooms connected by a narrow tunnel in the middle.
    This is a classic environment for studying spatial navigation between
    distinct compartments.
    
    Args:
        sr (float): Spatial resolution of the arena in units per pixel.
        **kwargs: Arena parameters including:
            room_width (float): Width of each room in spatial units. Defaults to 70.
            room_height (float): Height of rooms in spatial units. Defaults to 70.
            room_distance (float): Distance between rooms. Defaults to 30.
            tunnel_width (float): Width of connecting tunnel. Defaults to 20.
            vertical (bool): Whether to rotate the arena 90 degrees.
            
    Returns:
        np.ndarray: Arena map where 0 represents free space and 1 represents walls.
    """
    room_width = int(kwargs.get('room_width', 70) / sr) # pixel
    height = int(kwargs.get('room_height', 70) / sr) # pixel
    room_distance = int(kwargs.get('room_distance', 30) / sr) # pixel
    tunnel_width = int(kwargs.get('tunnel_width', 20) / sr) # pixel
    width = room_width*2 + room_distance
    border = 5

    arena_map = np.zeros((height, width))
    arena_map[0:(height-tunnel_width)//2, room_width:(room_width+room_distance)] = 1
    arena_map[-(height-tunnel_width)//2:, room_width:(room_width+room_distance)] = 1

    arena_map = np.pad(arena_map, border, 'constant', constant_values=1)

    if kwargs.get('vertical'):
        arena_map = np.rot90(arena_map, 1)

    return arena_map
