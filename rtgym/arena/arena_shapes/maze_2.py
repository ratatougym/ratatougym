import numpy as np

def generate_maze_2_arena(sr, **kwargs):
    """Generate a complex maze with rooms, tunnels, and dead ends.
    
    Creates an intricate maze with multiple rooms connected by tunnels,
    including dead ends and a circular path area. Designed for complex
    navigation experiments.
    
    Args:
        sr (float): Spatial resolution of the arena in units per pixel.
        **kwargs: Arena parameters (currently unused).
        
    Returns:
        np.ndarray: Arena map where 0 represents walls and 1 represents free space.
        
    Note:
        This maze uses inverted logic where 1 is free space and 0 is walls,
        unlike other arena generators.
    """
    width, height = 500, 400  # Maze dimensions
    border = 10  # Border size

    # Initialize with walls
    maze_map = np.zeros((height, width))

    # Define rooms and open spaces
    maze_map[20:100, 20:120] = 1   # Room 1
    maze_map[300:380, 20:200] = 1  # Room 2
    maze_map[200:280, 300:480] = 1 # Room 3
    maze_map[50:150, 350:450] = 1  # Room 4
    maze_map[150:250, 150:250] = 1 # Central Room

    # Tunnels and Paths
    maze_map[100:120, 60:400] = 1   # Horizontal Tunnel
    maze_map[50:300, 280:300] = 1   # Vertical Tunnel
    maze_map[120:150, 100:120] = 1  # Connecting Room 1 to Central Room
    maze_map[280:300, 400:480] = 1  # Connecting Room 3 to Room 4
    maze_map[150:200, 250:270] = 1  # Connecting Central Room to Vertical Tunnel

    # Dead Ends
    maze_map[20:50, 250:270] = 1    # Dead-end Path
    maze_map[350:380, 420:450] = 1  # Dead-end in Room 4

    # Strategic Walls
    # maze_map[100:300, 200:220] = 0  # Divide the left and right sections
    # maze_map[200:220, 120:300] = 0  # Horizontal wall in Central Room
    maze_map[150:180, 300:320] = 0  # Small divider in Room 4

    # Circular Path
    maze_map[320:350, 150:250] = 1  # Outer Circle
    maze_map[330:340, 160:240] = 0  # Inner Wall of Circle

    # Adding Border
    maze_map = np.pad(maze_map, border, mode='constant', constant_values=1)

    return maze_map
