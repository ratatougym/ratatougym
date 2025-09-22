import numpy as np

def generate_rectangle_arena(sr, **kwargs):
    """Generate a rectangular arena with a border around it.
    
    Creates a rectangular arena map where the interior is free space (0) and 
    the border is walls (1).
    
    Args:
        sr (float): Spatial resolution of the arena in units per pixel.
        **kwargs: Arena parameters including:
            dimensions (list): List of [width, height] dimensions in spatial units.
                              Defaults to [100, 100].
                              
    Returns:
        np.ndarray: Arena map where 0 represents free space and 1 represents walls.
        
    Raises:
        AssertionError: If dimensions is not a list of 2 elements.
    """
    dimensions = kwargs.get('dimensions', [100, 100])
    assert len(dimensions) == 2, 'dimensions must be a list of 2 elements'

    # convert dimensions from cm to pixel
    dimensions = np.array([int(dimensions[i] / sr) for i in range (2)]) # pixel

    border = 5
    arena_map = np.zeros(dimensions)
    arena_map = np.pad(arena_map, border, 'constant', constant_values=1)

    return arena_map
