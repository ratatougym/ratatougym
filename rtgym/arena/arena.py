import numpy as np
import matplotlib.pyplot as plt
from rtgym.utils import print_dict

from .arena_shapes import *

class Arena:
    """Arena environment for spatial navigation.
    
    The Arena defines the physical space where agents navigate, including
    walls, obstacles, and free space. It manages the spatial layout and
    provides utilities for position validation and random position generation.
    
    Args:
        gym (RatatouGym): Parent gym environment.
        **kwargs: Additional keyword arguments.
        
    Attributes:
        spatial_resolution (float): Spatial resolution in units per pixel.
        dimensions (tuple): Arena dimensions (height, width).
        free_space (np.ndarray): Coordinates of all free space positions.
        subscribers (list): List of objects subscribed to arena changes.
    """
    
    def __init__(self, gym, **kwargs):
        self.spatial_resolution = gym.spatial_resolution
        self._arena_map = None
        self.subscribers = []

    @property
    def arena_height(self):
        """Get arena height.
        
        Returns:
            int: Arena height in pixels.
        """
        return self.dimensions[0]

    @property
    def arena_width(self):
        """Get arena width.
        
        Returns:
            int: Arena width in pixels.
        """
        return self.dimensions[1]

    @property
    def map(self):
        """Get arena map (alias for arena_map).
        
        Returns:
            np.ndarray: Arena map where 0 is free space, 1 is wall.
        """
        return self.arena_map

    @property
    def arena_map(self):
        """Get the arena map.
        
        Returns:
            np.ndarray: Arena map where 0 represents free space and 1 represents walls.
        """
        return self._arena_map

    @property
    def inv_arena_map(self):
        """Get inverted arena map for visualization.
        
        Returns:
            np.ndarray: Inverted arena map where 1 is free space, 0 is wall.
        """
        return 1 - self.arena_map

    @arena_map.setter
    def arena_map(self, arena_map):
        """Set the arena map and update derived properties.
        
        Args:
            arena_map (np.ndarray): New arena map.
        """
        self._arena_map = arena_map
        self.dimensions = self._arena_map.shape
        self.free_space = np.argwhere(self._arena_map == 0)
        self.notify_subscribers()  # notify subscribers that the arena has changed

    def subscribe(self, subscriber):
        """Subscribe to arena change notifications.
        
        Args:
            subscriber (callable): Function to call when arena changes.
        """
        self.subscribers.append(subscriber)

    def notify_subscribers(self):
        """Notify all subscribers of arena changes.
        
        Calls all registered subscriber functions to inform them
        that the arena has been modified.
        """
        for subscriber in self.subscribers:
            subscriber()

    def set_arena_map(self, arena_map):
        """Set arena map with custom layout.
        
        Automatically pads edges with walls if they contain free space.
        
        Args:
            arena_map (np.ndarray): Arena map where 0 is free space, 1 is wall.
            
        Raises:
            AssertionError: If arena_map is not a numpy array.
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
        """Set the arena map with a predefined shape.

        Args:
            shape (str): Shape of the arena map.
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
        """ Get random positions in the arena """
        return self.free_space[np.random.choice(self.free_space.shape[0], size=size, replace=True)]

    def validate_index(self, pos):
        """ Check if the position is in the arena """
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
        """ Visualize arena """
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
        """ Get specs """
        not_include_list = ['arena', 'wt']
        params = {k: v for k, v in self.__dict__.items() if k not in not_include_list}
        return params

    def print_specs(self):
        """ 
        Print specs 
        
        Args:
            path (str): Path to the file where the arena map is saved.
        """
        print_dict(self.get_specs())

    def save(self, path):
        """ 
        Save arena map as an npz file
        
        Args:
            path (str): Path to the file where the arena map is saved.
        """
        np.savez(
            path, 
            arena_map=self.arena_map, 
            spatial_resolution=self.spatial_resolution,
        )

    def load(self, path):
        """ 
        Load arena map from an npz file

        Args:
            path (str): Path to the file where the arena map is saved.
        """
        data = np.load(path)
        self.spatial_resolution = data['spatial_resolution']
        self.set_arena_map(data['arena_map'])
