import numpy as np

def generate_cornered_rectangle_arena(sr, **kwargs):
    """Generate a rectangular arena with blocked corners.
    
    Creates a rectangular arena where the four corners are blocked by walls,
    creating a rounded rectangular navigation space. Useful for studying
    boundary effects and corner navigation strategies.
    
    Args:
        sr (float): Spatial resolution of the arena in units per pixel.
        **kwargs: Arena parameters including:
            width (float): Width of the rectangle in spatial units. Defaults to 100.
            height (float): Height of the rectangle in spatial units. Defaults to 100.
            corner (float): Size of corner blocks in spatial units. Defaults to 15.
            
    Returns:
        np.ndarray: Arena map where 0 represents free space and 1 represents walls.
        
    Raises:
        AssertionError: If corner size is too large for the arena dimensions.
    """
    width = int(kwargs.get('width', 100) / sr) # pixel
    height = int(kwargs.get('height', 100) / sr) # pixel
    corner = int(kwargs.get('corner', 15) / sr) # pixel

    assert 2*corner < width and 2*corner < height, 'corner must be smaller than half of the width and height'

    border = 5
    arena_map = np.zeros((height, width))
    arena_map[:corner, :corner] = 1
    arena_map[:corner, -corner:] = 1
    arena_map[-corner:, :corner] = 1
    arena_map[-corner:, -corner:] = 1
    arena_map = np.pad(arena_map, border, 'constant', constant_values=1)

    return arena_map
