import numpy as np
import torch
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
        self.gym = gym
        self.device = gym.device
        self._tensor_maps = {}
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
        if isinstance(arena_map, torch.Tensor):
            arena_map = arena_map.detach()
            arena_map = arena_map.cpu()
            arena_map = arena_map.numpy()
        if arena_map.ndim not in (2, 3):
            raise ValueError('Arena maps must have two or three dimensions.')
        self._tensor_maps.clear()
        self._arena_map = arena_map
        self.dimensions = self._arena_map.shape
        self.free_space_numpy = np.argwhere(self._arena_map == 0)
        self.notify_subscribers()  # notify subscribers that the arena has changed

    @property
    def ndim(self):
        return len(self.dimensions)

    def tensor_map(self, device='cpu'):
        """Cache a tensor view until the map is replaced through its setter."""
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            index = torch.cuda.current_device()
            device = torch.device('cuda', index)
        key = str(device)
        if key not in self._tensor_maps:
            arena_map = np.ascontiguousarray(self.arena_map)
            self._tensor_maps[key] = torch.as_tensor(arena_map, dtype=torch.float32, device=device)
        return self._tensor_maps[key]

    @property
    def map_(self):
        return self.tensor_map(self.gym.device)

    @map_.setter
    def map_(self, value):
        self.arena_map = value

    @property
    def invmap_(self):
        return 1 - self.map_

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
        if isinstance(arena_map, torch.Tensor):
            arena_map = arena_map.detach()
            arena_map = arena_map.cpu()
            arena_map = arena_map.numpy()
        if not isinstance(arena_map, np.ndarray) or arena_map.ndim not in (2, 3):
            raise ValueError('arena_map must be a 2D or 3D array.')

        # Preserve official padding: add a wall only when a whole face is free.
        for axis in range(arena_map.ndim):
            for side in (0, -1):
                face = np.take(arena_map, side, axis=axis)
                free_face = np.all(face == 0)
                if free_face:
                    padding = [(0, 0)] * arena_map.ndim
                    padding[axis] = (1, 0) if side == 0 else (0, 1)
                    arena_map = np.pad(arena_map, padding, mode='constant', constant_values=1)

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
            'trainer_0': generate_trainer_0_arena,
            'box': generate_box_arena,
            'hairpin': generate_hairpin_arena,
            'carpenter_rooms': generate_carpenter_rooms_arena
        }

        if shape not in shape_generators:
            raise ValueError(f"Unknown shape '{shape}'. Valid options are: {list(shape_generators.keys())}")

        # Generate the arena map
        self.arena_map = shape_generators[shape](self.spatial_resolution, **kwargs)

    @property
    def free_space(self):
        return torch.as_tensor(self.free_space_numpy, device=self.device)

    def generate_random_pos(self, batch_size):
        """Sample free positions as a tensor on the arena device."""
        n_free = len(self.free_space_numpy)
        indices = torch.randint(n_free, (batch_size,), device=self.device)
        return self.free_space[indices]

    def generate_random_pos_numpy(self, size):
        """Sample with the original NumPy RNG for the random_walk control."""
        count = len(self.free_space_numpy)
        indices = np.random.choice(count, size=size, replace=True)
        return self.free_space_numpy[indices]

    def validate_index(self, pos):
        """ Check if the position is in the arena """
        if isinstance(pos, torch.Tensor):
            return self._validate_tensor_index(pos)
        if len(pos.shape) == 1:
            pos = pos[np.newaxis, :]
        # check dimension
        assert pos.shape[1] == self.ndim, "pos must match the arena dimension"
        # check if the indices are defined
        is_negative = np.all(pos >= 0, axis=1)
        is_exceed = np.all(pos < self.dimensions, axis=1)
        valid_idx = np.logical_and(is_negative, is_exceed)
        is_wall = np.full(pos.shape[0], True)
        is_wall[valid_idx] = self.arena_map[tuple(pos[valid_idx].T)] == 1
        return np.logical_not(is_wall)

    def _validate_tensor_index(self, pos):
        pos = pos.reshape(-1, self.ndim)
        in_bounds = torch.ones(pos.shape[0], dtype=torch.bool, device=pos.device)
        indices = []
        for axis, size in enumerate(self.dimensions):
            index = pos[:, axis]
            in_bounds = in_bounds & (index >= 0) & (index < size)
            index = index.clamp(0, size - 1)
            indices.append(index)
        indices = tuple(indices)
        arena_map = self.tensor_map(pos.device)
        free = arena_map[indices] == 0
        return in_bounds & free

    def vis(self):
        """ Visualize arena """
        if self.ndim != 2:
            raise ValueError('Arena.vis requires a 2D map; plot a slice for 3D.')
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
