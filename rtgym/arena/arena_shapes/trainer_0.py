import numpy as np

def generate_trainer_0_arena(sr, **kwargs):
    # Extract parameters with default values
    max_radius = kwargs.get('max_radius', 100)  # in physical units
    min_radius = kwargs.get('min_radius', 50)  # in physical units
    n_rooms = kwargs.get('n_rooms', 3)
    border = kwargs.get('border', 5)  # in pixels

    # Validate parameters
    if n_rooms < 1:
        raise ValueError("n_rooms must be at least 1.")
    if min_radius > max_radius:
        raise ValueError("min_radius must be less than or equal to max_radius.")
    
    # Calculate radii in pixels, evenly spaced between max_radius and min_radius
    radii_physical = np.linspace(max_radius, min_radius, n_rooms)
    radii = (radii_physical / sr).astype(int)  # Convert to pixels

    # Calculate total width: sum of diameters + borders
    total_diameter = np.sum(2 * radii)
    total_border = (n_rooms + 1) * border
    width = total_diameter + total_border
    height = 2 * radii.max() + 2 * border  # Height accommodates the largest circle

    arena_map = np.ones((height, width), dtype=np.uint8)  # Initialize all as walls

    # Initialize starting x position
    current_x = border

    # Center y position
    center_y = height // 2

    for idx, r in enumerate(radii):
        # Define the center for the current circle
        center_x = current_x + r
        # Create a grid for the arena
        y, x = np.ogrid[:height, :width]
        dist_sq = (x - center_x)**2 + (y - center_y)**2
        # Define the circle area
        arena_map[dist_sq <= r**2] = 0  # Free space
        # Move to the next position: current_x += diameter + border
        current_x += 2 * r + border

    return arena_map
