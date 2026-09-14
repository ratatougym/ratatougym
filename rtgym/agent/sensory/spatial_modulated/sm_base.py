import numpy as np
import torch
import pickle
from numpy.random import default_rng
from typing import Union
import matplotlib.pyplot as plt
import rtgym
import rtgym.utils as utils
from rtgym.dataclass import AgentState, Trajectory
from rtgym.utils.common import hash_seed

class SMBase():
    """Base class for spatially modulated sensory cells.
    
    Provides common functionality for all spatial sensory modalities such as
    place cells, grid cells, and boundary cells. These cells respond based
    on the agent's position in space.
    
    Args:
        arena (Arena): Arena environment object.
        n_cells (int): Number of cells in this sensory modality.
        sensory_key (str): Unique identifier for this sensory type.
        seed (int, optional): Random seed for reproducible cell generation.
        **kwargs: Additional keyword arguments.
        
    Attributes:
        arena (Arena): Arena environment.
        n_cells (int): Number of sensory cells.
        sensory_key (str): Unique sensory identifier.
        rng (np.random.Generator): Random number generator.
        response_map (np.ndarray): Spatial response map for all cells.
        
    Raises:
        AssertionError: If instantiated directly (abstract class) or n_cells <= 0.
    """
    
    sens_category = 'spatial_modulated'
    sens_type = 'sm_base'
    
    def __init__(self, arena, n_cells, sensory_key, seed=None, **kwargs):
        self.arena = arena
        self.n_cells = n_cells
        self.sensory_key = sensory_key
        self._tensor_response_map = None

        # Initialize random number generator
        if seed is not None:
            seed = hash_seed(seed, sensory_key)
        self.rng = default_rng(seed)

        # check if the class is the base class
        assert type(self) != SMBase, "SMBase is an abstract class"
        assert self.n_cells > 0, "n_cells <= 0"


    def _init_response_map(self):
        """Initialize the spatial response map for all cells.
        
        Creates a zero-filled response map with shape (n_cells, height, width).
        Border padding is included in the response field.
        """
        self.response_map = np.zeros((self.n_cells, *self.arena.dimensions))

    def get_specs(self):
        """Get specifications of this sensory modality.
        
        Returns:
            dict: Dictionary containing sensory specifications including
                  number of cells and response field dimensions.
        """
        return {
            'n_cells': self.n_cells,
            'response_field_width (with 5 pixels padding)': self.response_map.shape[1],
            'response_field_height (with 5 pixels padding)': self.response_map.shape[2],
        }

    def print_specs(self):
        """Print specifications of this sensory modality.
        
        Prints the specifications returned by get_specs() in a formatted manner.
        """
        utils.print_dict(self.get_specs())

    def to(self, device, dtype=None):
        """Cache a tensor response field. Call again after editing response_map."""
        if dtype is None:
            if isinstance(self.response_map, torch.Tensor):
                dtype = self.response_map.dtype
            else:
                dtype = torch.float32
        response_map = self.response_map
        if isinstance(response_map, np.ndarray):
            response_map = np.ascontiguousarray(response_map)
        self._tensor_response_map = torch.as_tensor(response_map, dtype=dtype, device=device)
        return self

    def get_response(self, agent_data, return_format=None, device=None):
        """Look up states or trajectories on CPU or from a cached tensor field.

        NumPy input keeps the original NumPy result unless tensor output is
        requested explicitly. Tensor input uses its own device by default.
        """
        if not isinstance(agent_data, (AgentState, Trajectory)):
            raise TypeError('agent_data must be an AgentState or Trajectory.')
        coord = agent_data.int_coord
        tensor_input = isinstance(coord, torch.Tensor)
        if return_format is None:
            return_format = 'tensor' if tensor_input else 'array'
        if return_format not in ('array', 'tensor'):
            raise ValueError('return_format must be array or tensor.')

        # Keep the legacy NumPy lookup and dtype unchanged.
        numpy_field = isinstance(self.response_map, np.ndarray)
        if return_format == 'array' and numpy_field:
            if tensor_input:
                coord = coord.detach()
                coord = coord.cpu()
                coord = coord.numpy()
            indices = tuple(coord[..., axis] for axis in range(self.arena.ndim))
            response = self.response_map[(slice(None), *indices)]
            return np.moveaxis(response, 0, -1)

        # Move coordinates, not the whole field, for repeated device queries.
        if device is None:
            if tensor_input:
                device = coord.device
            elif self._tensor_response_map is not None:
                device = self._tensor_response_map.device
            elif isinstance(self.response_map, torch.Tensor):
                device = self.response_map.device
            else:
                device = 'cpu'
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            index = torch.cuda.current_device()
            device = torch.device('cuda', index)
        cached = self._tensor_response_map
        if cached is None or cached.device != device:
            self.to(device)
        coord = torch.as_tensor(coord, device=device, dtype=torch.long)
        indices = tuple(coord[..., axis] for axis in range(self.arena.ndim))
        response = self._tensor_response_map[(slice(None), *indices)]
        response = torch.movedim(response, 0, -1)
        if return_format == 'array':
            response = response.detach()
            response = response.cpu()
            return response.numpy()
        return response

    def vis(self, N=10, cmap='jet', *args, **kwargs):
        """Visualize the spatially modulated cells.
        
        Args:
            N (int): Number of cells to visualize.
            cmap (str): Colormap for visualization.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            matplotlib figure: Visualization of the first N cells.
        """
        cells = self.response_map[:N]
        return utils.visualize_fields(cells, cmap=cmap, mask=self.arena.inv_arena_map)

    def save(self, file_path):
        """
        Save the object to a file, excluding dynamically generated data.
        
        Args:
            file_path (str): Path to the file where the object will be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self.state_dict(), f)


    @classmethod
    def load(cls, file_path, arena):
        """
        Load the GridCell object from a file and reconstruct it.

        Args:
            file_path (str): Path to the saved file.
            arena (Arena): Arena object to reinitialize the class.

        Returns:
            Reconstructed GridCell object.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        return cls.load_from_dict(data, arena)
