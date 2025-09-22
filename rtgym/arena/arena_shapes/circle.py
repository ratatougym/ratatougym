import numpy as np

def generate_circle_arena(sr, **kwargs):
    """Generate a circular arena with a border around it.
    
    Creates a circular arena map where the interior is free space (0) and 
    the exterior and border are walls (1).
    
    Args:
        sr (float): Spatial resolution of the arena in units per pixel.
        **kwargs: Arena parameters including:
            radius (float): Radius of the circle in spatial units. Defaults to 50.
            
    Returns:
        np.ndarray: Arena map where 0 represents free space and 1 represents walls.
    """
    radius = int(kwargs.get('radius', 50) / sr) # pixel
    border = 5

    width = radius*2 + border*2
    arena_map = np.ones((width, width)) # Fill everything as wall

    # generate a circle
    for i in range(arena_map.shape[0]):
        for j in range(arena_map.shape[1]):
            if (i-radius-border)**2 + (j-radius-border)**2 <= radius**2:
                arena_map[i, j] = 0 # Cut out the circle

    return arena_map
