import numpy as np
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
from numpy.random import default_rng

from rtgym.utils import print_dict
from rtgym.utils.common import hash_seed


class MMBase():
    """Base class for movement modulated sensory cells.
    
    Provides common functionality for sensory modalities that respond to
    movement parameters such as speed, direction, and acceleration.
    These cells are temporally modulated based on agent movement.

    Args:
        arena (Arena): Arena environment object.
        n_cells (int): Number of cells in this sensory modality.
        t_res (float): Temporal resolution in milliseconds.
        sensory_key (str): Unique identifier for this sensory type.
        seed (int, optional): Random seed for reproducible cell generation.
        **kwargs: Additional keyword arguments including:
            magnitude (float): Base magnitude for cell responses.
            msigma (float): Standard deviation for magnitude variation.
            
    Attributes:
        arena (Arena): Arena environment.
        t_res (float): Temporal resolution.
        sensory_key (str): Unique sensory identifier.
        n_cells (int): Number of sensory cells.
        magnitude (np.ndarray): Magnitude values for each cell.
        msigma (float): Magnitude standard deviation.
        rng (np.random.Generator): Random number generator.
        
    Raises:
        AssertionError: If instantiated directly (abstract class) or n_cells <= 0.
    """
    
    sens_type = 'mm_base'
    
    def __init__(self, arena, n_cells, t_res, sensory_key, seed=None, **kwargs):
        self.arena = arena
        self.t_res = t_res  # temporal resolution
        self.sensory_key = sensory_key
        self.n_cells = n_cells
        self.magnitude = kwargs.get('magnitude', None)
        self.msigma = kwargs.get('msigma', 0)

        # Initialize random number generator
        if seed is not None:
            seed = hash_seed(seed, sensory_key)
        self.rng = default_rng(seed)

        if self.magnitude is not None:
            if self.msigma >= 0:
                # make the magnitude to be a vector of length n_cells
                self.magnitude = np.clip(self.rng.normal(self.magnitude, self.msigma, self.n_cells), a_min=0, a_max=None)
                self.magnitude = self.magnitude.reshape(1, 1, self.n_cells)

        # check if the class is the base class
        assert type(self) != MMBase, "MMBase is an abstract class"
        assert self.n_cells > 0, "n_cells <= 0"


    def _smooth_res(self, res):
        """Smooth input with a causal Gaussian-like kernel.
        
        Applies temporal smoothing to movement modulated responses using
        a one-sided Gaussian kernel for causal filtering.
        
        Args:
            res (np.ndarray): Input responses of shape (n_batch, n_time, n_cells).
            
        Returns:
            np.ndarray: Smoothed responses of the same shape as input.
        """
        if self.sigma_ts == 0:
            return_res = res
        else:
            n = res.shape[2]
            sigs = np.clip(np.random.normal(self.sigma_ts, self.ssigma_ts, n), a_min=5e-2, a_max=None)
            return_res = np.empty_like(res)
            
            for i in range(n):
                # Constructing a one-sided Gaussian-like kernel
                half_size = int(3 * sigs[i])  # To approximate a Gaussian
                kernel = np.exp(-np.arange(half_size + 1) ** 2 / (2 * sigs[i] ** 2))
                kernel = kernel / kernel.sum()  # Normalize to make it a probability distribution

                # Convolve with the kernel along the time dimension with 'valid' padding
                filtered = convolve1d(res[:, :-1, i], weights=kernel, axis=1, mode='nearest')
                return_res[:, :-1, i] = filtered  # The last time step is 0.
                return_res[:, -1, i] = res[:, -1, i]  # Pad the last time step with the last value
            
        # Scale the responses if magnitude is not None
        if self.magnitude is not None:
            return_res *= self.magnitude
        
        return return_res

    def _t_to_ts(self, t):
        """Convert time in seconds to timesteps.
        
        Args:
            t (float): Time value in seconds.
            
        Returns:
            int: Corresponding timestep.
        """
        return int(t*1e3/self.t_res)
    
    def _duplicate_res(self, res, divisor=1):
        """Duplicate input array to match the number of cells.
        
        Tiles the input array along the last dimension to create responses
        for all cells in this sensory modality.
        
        Args:
            res (np.ndarray): Input array of shape (n_batch, n_time, n_features).
            divisor (int): Divisor for n_cells (default 1, use 2 for displacement/acceleration).
            
        Returns:
            np.ndarray: Duplicated array of shape (n_batch, n_time, n_cells//divisor).
        """
        return np.tile(res, (1, 1, self.n_cells//divisor))

    def vis(self, traj, N, *args, **kwargs):
        """Visualize the movement modulated cells.
        
        Args:
            traj (Trajectory): Trajectory data for visualization.
            N (int): Number of cells to visualize.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            tuple: Matplotlib figure and axis objects.
        """
        responses = self.get_responses(traj)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(responses[0, :-1, :N]) # Last dimension will be 0.
        ax.set_title(self.__class__.__name__)
        ax.set_xlabel('Time')
        ax.set_ylabel('Response')
        return fig, ax

    def get_specs(self):
        """Get specifications of this sensory modality.
        
        Returns:
            dict: Dictionary containing sensory specifications.
        """
        return {
            'n_cells': self.n_cells,
        }
    
    def print_specs(self):
        """Print specifications of this sensory modality.
        
        Prints the specifications returned by get_specs() in a formatted manner.
        """
        print_dict(self.get_specs())
