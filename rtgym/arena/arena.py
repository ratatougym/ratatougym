import numpy as np
import matplotlib.pyplot as plt
from rtgym.utils import print_dict

from .arena_shapes import *

class Arena:
    """Arena environment for spatial navigation and agent behavior simulation.
    
    The Arena class defines the physical space where agents navigate, including
    walls, obstacles, and free space. It manages the spatial layout and
    provides utilities for position validation, random position generation,
    and arena visualization. The arena uses a binary map representation where
    0 indicates free space and 1 indicates walls or obstacles.
    
    The arena supports various predefined shapes (rectangle, circle, maze, etc.)
    and custom layouts. It also implements a subscription system to notify
    other components when the arena configuration changes.
    
    Args:
        gym (RatatouGym): Parent gym environment that provides spatial resolution
            and other global parameters.
        **kwargs: Additional keyword arguments for arena configuration.
        
    Attributes:
        spatial_resolution (float): Spatial resolution in units per pixel, determining
            the real-world scale of the arena.
        dimensions (tuple): Arena dimensions as (height, width) in pixels.
        free_space (np.ndarray): Array of coordinates representing all valid
            free space positions where agents can move.
        subscribers (list): List of callable objects subscribed to arena change
            notifications for automatic updates when arena layout changes.
    """
    
    def __init__(self, gym, **kwargs):
        self.spatial_resolution = gym.spatial_resolution
        self._arena_map = None
        self.subscribers = []

    @property
    def arena_height(self):
        """
        Get arena height in pixels.
        
        Returns the vertical dimension of the arena map. This represents
        the number of pixels along the height axis of the arena.
        
        Returns:
            int: Arena height in pixels (number of rows in the arena map).
        """
        return self.dimensions[0]

    @property
    def arena_width(self):
        """
        Get arena width in pixels.
        
        Returns the horizontal dimension of the arena map. This represents
        the number of pixels along the width axis of the arena.
        
        Returns:
            int: Arena width in pixels (number of columns in the arena map).
        """
        return self.dimensions[1]

    @property
    def map(self):
        """
        Get arena map (convenient alias for arena_map property).
        
        Provides a shorter alias to access the arena map for convenience
        in code that frequently references the map.
        
        Returns:
            np.ndarray: Arena map where 0 represents free space and 1 represents walls.
        """
        return self.arena_map

    @property
    def arena_map(self):
        """
        Get the binary arena map representation.
        
        Returns the internal arena map as a 2D numpy array where each pixel
        represents a spatial location. The binary encoding uses 0 for free
        space (where agents can move) and 1 for walls or obstacles.
        
        Returns:
            np.ndarray: Arena map where 0 represents free space and 1 represents walls.
                Shape is (height, width) corresponding to the arena dimensions.
        """
        return self._arena_map

    @property
    def inv_arena_map(self):
        """
        Get inverted arena map for visualization purposes.
        
        Returns a version of the arena map with inverted values (1 becomes 0, 0 becomes 1).
        This is primarily used for visualization where free space should appear bright
        and walls should appear dark, which is more intuitive for display.
        
        Returns:
            np.ndarray: Inverted arena map where 1 represents free space and 0 represents walls.
                Useful for matplotlib visualization with standard colormaps.
        """
        return 1 - self.arena_map

    @arena_map.setter
    def arena_map(self, arena_map):
        """
        Set the arena map and automatically update derived properties.
        
        When a new arena map is assigned, this setter automatically updates
        all dependent properties including dimensions and free space coordinates.
        It also notifies all subscribers of the change to maintain consistency
        across the system.
        
        Args:
            arena_map (np.ndarray): New arena map where 0 is free space and 1 is wall.
                Must be a 2D numpy array with binary values.
        """
        self._arena_map = arena_map
        self.dimensions = self._arena_map.shape
        self.free_space = np.argwhere(self._arena_map == 0)
        self.notify_subscribers()  # notify subscribers that the arena has changed

    def subscribe(self, subscriber):
        """
        Subscribe to arena change notifications for automatic updates.
        
        Adds a callable object to the subscriber list. Subscribers will be
        automatically notified whenever the arena configuration changes,
        allowing dependent systems (like behavior models or sensory systems)
        to update themselves accordingly.
        
        Args:
            subscriber (callable): Function or method to call when arena changes.
                Should accept no arguments and handle arena updates internally.
        """
        self.subscribers.append(subscriber)

    def notify_subscribers(self):
        """
        Notify all subscribers of arena changes.
        
        Calls all registered subscriber functions to inform them that the arena
        has been modified. This enables automatic propagation of arena changes
        throughout the system, ensuring that all dependent components stay
        synchronized with the current arena state.
        
        Note:
            Subscribers should be designed to handle exceptions gracefully
            to prevent one failing subscriber from blocking others.
        """
        for subscriber in self.subscribers:
            subscriber()

    def set_arena_map(self, arena_map):
        """
        Set arena map with custom layout and automatic boundary padding.
        
        Accepts a custom arena map and ensures it has proper boundary conditions
        by automatically padding edges with walls if they contain free space.
        This prevents agents from escaping the arena boundaries. The method
        treats the input as a binary matrix where 0 represents free space
        and 1 represents walls.
        
        Args:
            arena_map (np.ndarray): Custom arena map where 0 is free space and 1 is wall.
                Must be a 2D numpy array with binary values.
            
        Raises:
            AssertionError: If arena_map is not a numpy array.
            
        Note:
            The method automatically adds wall padding to any edge that contains
            only free space (all zeros) to ensure the arena is properly enclosed.
        """
        assert isinstance(arena_map, np.ndarray), "arena map must be a numpy array"

        # Check if its edges are all 1. If not, pad them with 1
        if np.all(arena_map[0, :] == 0):
            arena_map = np.pad(arena_map, ((1, 0), (0, 0)), mode='constant', constant_values=1)
        if np.all(arena_map[-1, :] == 0):
            arena_map = np.pad(arena_map, ((0, 1), (0, 0)), mode='constant', constant_values=1)
        if np.all(arena_map[:, 0] == 0):
            arena_map = np.pad(arena_map, ((0, 0), (1, 0)), mode='constant', constant_values=1)
        if np.all(arena_map[:, -1] == 0):
            arena_map = np.pad(arena_map, ((0, 0), (0, 1)), mode='constant', constant_values=1)

        self.arena_map = arena_map

    def init_arena_map(self, shape, **kwargs):
        """
        Initialize the arena map with a predefined geometric shape.
        
        Generates an arena map using one of several predefined shape templates.
        Each shape generator creates an appropriate binary map based on the
        spatial resolution and any additional parameters provided. This method
        provides a convenient way to create standard arena configurations
        without manually defining the map.

        Args:
            shape (str): Name of the predefined shape to generate. Available options
                include 'rectangle', 'circle', 'triangle', 'two_rooms', 
                'cornered_rectangle', 'maze_0', 'maze_1', 'maze_2', and 'trainer_0'.
            **kwargs: Additional keyword arguments passed to the shape generation
                function. These vary by shape type and may include dimensions,
                room configurations, maze complexity, etc.
                
        Raises:
            ValueError: If the specified shape name is not recognized.
            
        Example:
            arena.init_arena_map('rectangle', width=100, height=80)
            arena.init_arena_map('circle', radius=50)
            arena.init_arena_map('maze_0', complexity=0.3)
        """
        # Map shape names to their corresponding generation functions
        shape_generators = {
            'rectangle': generate_rectangle_arena,
            'circle': generate_circle_arena,
            'triangle': generate_triangle_arena,
            'two_rooms': generate_two_rooms_arena,
            'cornered_rectangle': generate_cornered_rectangle_arena,
            'maze_0': generate_maze_0_arena,
            'maze_1': generate_maze_1_arena,
            'maze_2': generate_maze_2_arena,
            'trainer_0': generate_trainer_0_arena
        }

        if shape not in shape_generators:
            raise ValueError(f"Unknown shape '{shape}'. Valid options are: {list(shape_generators.keys())}")

        # Generate the arena map
        self.arena_map = shape_generators[shape](self.spatial_resolution, **kwargs)

    def generate_random_pos(self, size):
        """
        Generate random valid positions within the arena free space.
        
        Selects random positions from all available free space coordinates
        in the arena. This is useful for initializing agent positions,
        placing objects, or generating random waypoints for navigation tasks.
        Positions are guaranteed to be in valid free space areas.
        
        Args:
            size (int): Number of random positions to generate. Each position
                will be selected independently with replacement from available
                free space coordinates.
                
        Returns:
            np.ndarray: Array of shape (size, 2) containing random coordinates
                in the format [row, column] corresponding to valid positions
                in the arena free space.
                
        Note:
            Sampling is done with replacement, so duplicate positions are possible
            if size is large relative to available free space.
        """
        return self.free_space[np.random.choice(self.free_space.shape[0], size=size, replace=True)]

    def validate_index(self, pos):
        """
        Validate whether given positions are within arena bounds and free space.
        
        Checks if the provided positions are valid locations within the arena
        by verifying they are within bounds and not located in walls or obstacles.
        This method handles both single positions and arrays of positions efficiently.
        
        Args:
            pos (np.ndarray): Position(s) to validate. Can be a single position
                as a 1D array [row, col] or multiple positions as a 2D array
                of shape (n_positions, 2).
                
        Returns:
            np.ndarray: Boolean array indicating validity of each position.
                True means the position is valid (within bounds and in free space),
                False means invalid (out of bounds or in a wall).
                
        Raises:
            AssertionError: If pos does not have exactly 2 columns (x, y coordinates).
            
        Note:
            The method automatically handles dimension conversion for single positions
            and efficiently processes multiple positions using vectorized operations.
        """
        if len(pos.shape) == 1:
            pos = pos[np.newaxis, :]
        # check dimension
        assert pos.shape[1] == 2, "pos must be a 2D array"
        # check if the indices are defined
        is_negative = np.all(pos >= 0, axis=1)
        is_exceed = np.all(pos < self.dimensions, axis=1)
        valid_idx = np.logical_and(is_negative, is_exceed)
        is_wall = np.full(pos.shape[0], True)
        is_wall[valid_idx] = self.arena_map[tuple(pos[valid_idx].T)] == 1
        return np.logical_not(is_wall)

    def vis(self):
        """
        Visualize the arena layout with proper scaling and labels.
        
        Creates a matplotlib visualization of the arena showing walls and free space
        with appropriate scaling based on spatial resolution. The visualization
        includes axis labels in real-world units, a legend distinguishing walls
        from free space, and excludes border pixels from size calculations for
        more accurate representation.
        
        Returns:
            tuple: A tuple containing (figure, axis) objects from matplotlib
                for further customization or saving of the visualization.
                
        Note:
            The visualization uses inverted arena map for better visual contrast
            and automatically scales axis labels to show real-world distances
            based on the spatial resolution. Border pixels (5 on each side)
            are excluded from the displayed dimensions.
        """
        fig, ax = plt.subplots()
        ax.imshow(self.inv_arena_map, vmin=-1, vmax=1, cmap='gray')
        # ax.axis('off')
        # plot two bar legend indicating the wall and the free space
        ax.bar([0, 0], [0, 0], color='#888', label='wall')
        ax.bar([0, 0], [0, 0], color='#eee', label='free space')
        ax.set_title(rf'Arena (size={self.dimensions[1]-10}x{self.dimensions[0]-10} $pixels^2$, excluding border)')
        ax.set_xticks(np.linspace(5, self.dimensions[1]-5, 5))
        ax.set_xticklabels(np.linspace(0, self.dimensions[1]-10, 5)*self.spatial_resolution)
        ax.set_yticks(np.linspace(5, self.dimensions[0]-5, 5))
        ax.set_yticklabels(np.linspace(0, self.dimensions[0]-10, 5)*self.spatial_resolution)
        ax.set_xlabel('width (cm)')
        ax.set_ylabel('height (cm)')
        ax.legend()
        return fig, ax

    def get_specs(self):
        """
        Get arena specification parameters as a dictionary.
        
        Collects and returns all relevant arena parameters and configuration
        settings as a dictionary. This excludes certain internal objects
        that are not serializable or relevant for specification purposes.
        Useful for saving configurations, debugging, or recreating arenas.
        
        Returns:
            dict: Dictionary containing arena specifications including
                spatial resolution, dimensions, and other configuration
                parameters. Excludes non-serializable objects like 'arena'
                and 'wt' references.
        """
        not_include_list = ['arena', 'wt']
        params = {k: v for k, v in self.__dict__.items() if k not in not_include_list}
        return params

    def print_specs(self):
        """ 
        Print arena specifications in a formatted, human-readable way.
        
        Displays all arena configuration parameters and settings using
        a formatted print utility. This provides a convenient way to
        inspect the current arena configuration during development,
        debugging, or documentation purposes.
        
        Note:
            The output format is determined by the print_dict utility
            function, which typically provides nice formatting with
            proper indentation and type information.
        """
        print_dict(self.get_specs())

    def save(self, path):
        """ 
        Save arena configuration to a compressed numpy archive file.
        
        Serializes the arena map and spatial resolution to an NPZ file
        for later loading. This enables persistence of arena configurations
        across sessions and sharing of arena setups between different
        experiments or users.
        
        Args:
            path (str): File path where the arena data should be saved.
                Should end with '.npz' extension for clarity, though
                numpy will handle the format automatically.
                
        Note:
            Only essential data (arena_map and spatial_resolution) is saved
            to keep file size minimal. Other derived properties will be
            reconstructed when the arena is loaded.
        """
        np.savez(
            path, 
            arena_map=self.arena_map, 
            spatial_resolution=self.spatial_resolution,
        )

    def load(self, path):
        """ 
        Load arena configuration from a previously saved numpy archive file.
        
        Restores arena state from an NPZ file created by the save method.
        This reconstructs the arena map, spatial resolution, and all derived
        properties like dimensions and free space coordinates. The loaded
        arena will be functionally identical to the original saved arena.

        Args:
            path (str): Path to the NPZ file containing saved arena data.
                Must be a file created by the Arena.save() method or
                containing compatible 'arena_map' and 'spatial_resolution' arrays.
                
        Raises:
            FileNotFoundError: If the specified file does not exist.
            KeyError: If the file does not contain required data fields.
            
        Note:
            Loading will trigger subscriber notifications as the arena map
            is updated, ensuring all dependent systems are properly updated.
        """
        data = np.load(path)
        self.spatial_resolution = data['spatial_resolution']
        self.set_arena_map(data['arena_map'])
