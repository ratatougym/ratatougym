import numpy as np
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
    
    neuron_category = 'spatial_modulated'
    neuron_type = 'sm_base'
    
    def __init__(self, arena, n_cells, sensory_key, seed=None, **kwargs):
        self.arena = arena
        self.n_cells = n_cells
        self.sensory_key = sensory_key

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

    def get_response(self, agent_data: Union[AgentState, Trajectory]):
        """Get sensory responses for given agent data.
        
        Args:
            agent_data (AgentState or Trajectory): Agent data containing coordinates.
            
        Returns:
            np.ndarray: Sensory responses with shape:
                - For Trajectory: (n_batch, n_timesteps, n_cells)
                - For AgentState: (n_batch, n_cells)
                
        Raises:
            ValueError: If agent_data type is not supported.
        """
        if isinstance(agent_data, Trajectory):
            return self.response_map[:, agent_data.int_coord[..., 0], agent_data.int_coord[..., 1]].transpose(1, 2, 0)
        elif isinstance(agent_data, AgentState):
            return self.response_map[:, agent_data.int_coord[:, 0], agent_data.int_coord[:, 1]].transpose(1, 0)
        else:
            raise ValueError(f"Invalid agent_data type: {type(agent_data)}, must be rtgym.dataclass.Trajectory or rtgym.dataclass.AgentState")

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
