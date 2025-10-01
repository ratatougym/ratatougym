import numpy as np
from .sm_base import SMBase
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class PlaceCell(SMBase):
    """Place cell sensory responses.
    
    Place cells are neurons that fire when an animal is in specific locations in an environment.
    Each place cell has a "place field" - a region where it fires maximally. The firing rate 
    typically follows a Gaussian distribution centered on the place field, decreasing with 
    distance from the center.
    
    Place cells are a key component of the brain's spatial navigation system, discovered by 
    O'Keefe and Dostrovsky in 1971. They are primarily found in the hippocampus and are 
    thought to form a cognitive map of the environment.
    
    This implementation models place cells with Gaussian or difference-of-Gaussian tuning curves.
    The difference-of-Gaussian option creates more realistic place fields with inhibitory surrounds.
    
    Args:
        arena (Arena): Arena environment object defining the navigation space.
        **kwargs: Additional keyword arguments including:
            n_cells (int): Number of place cells to create.
            sigma (float or list): Width of place fields in spatial units. If float, all cells
                use same sigma. If list, each cell gets its own sigma. Defaults to 8.
            ssigma (float): Standard deviation of random variation in place field sizes.
                Only used when sigma is float. Defaults to 0.
            dg_ratio (float): Ratio between center and surround Gaussian for 
                difference-of-Gaussian fields. Values > 1 create inhibitory surrounds.
                Defaults to 1 (simple Gaussian).
            magnitude (float): Scaling factor for response amplitudes. Defaults to None.
            normalize (bool): Whether to normalize responses to [0,1]. Defaults to False.
            
    Attributes:
        neuron_type (str): Sensory type identifier 'place_cell'.
        response_map (np.ndarray): Spatial response map of shape (n_cells, *arena_dimensions).
            Contains the firing rate map for each cell across the entire arena.
    """
    
    neuron_type = 'place_cell'
    
    def __init__(self, arena, **kwargs):
        super().__init__(arena, **kwargs)

        # parameters
        self.sigma = kwargs.get('sigma', 8)  # Place field width
        self.ssigma = kwargs.get('ssigma', 0)  # Random variation in field width
        self.dg_ratio = kwargs.get('dg_ratio', 1)  # For difference-of-Gaussian fields
        self.magnitude = kwargs.get('magnitude', None)  # Response scaling
        self.normalize = kwargs.get('normalize', False)  # Normalize to [0,1]

        # check parameters and initialize responses
        self._check_params()
        self._init_params()
        self._init_response_map()

    def _check_params(self):
        """ 
        Validate initialization parameters.
        
        Raises:
            AssertionError: If parameters are invalid (e.g. negative values).
        """
        assert self.n_cells > 0, "n_cells <= 0"
        # sigma can be a list or an integer
        if isinstance(self.sigma, int):
            assert self.sigma > 0, "sigma <= 0"
        elif isinstance(self.sigma, list):
            for s in self.sigma:
                assert s > 0, "sigma <= 0"

    def _init_params(self):
        """
        Convert parameters from real units to grid units using arena resolution.
        """
        if isinstance(self.sigma, int):
            self.sigma = int(self.sigma/self.arena.spatial_resolution)
            self.ssimga = int(self.ssigma/self.arena.spatial_resolution)
        elif isinstance(self.sigma, list):
            self.sigma = [int(s/self.arena.spatial_resolution) for s in self.sigma]
            if self.ssigma != 0:
                print("Warning: ssigma is not used when sigma is a list")

    def _init_response_map(self):
        """
        Generate the spatial response fields for all place cells.
        
        Creates Gaussian or difference-of-Gaussian response fields centered at random
        locations in the arena's free space. The response map contains the firing rate
        for each cell at every location in the arena.
        """
        super()._init_response_map()

        arena_dims = self.arena.dimensions
        arena_free_space = self.arena.free_space
        # Randomly select cell centers from available free space
        place_centers = arena_free_space[self.rng.choice(arena_free_space.shape[0], size=self.n_cells, replace=False)]
        cells = np.zeros((self.n_cells, *arena_dims))
        # Generate place field sizes, either constant or with random variation
        sigma_list = self.rng.normal(self.sigma, self.ssigma, self.n_cells) if self.ssigma != 0 else [self.sigma] * self.n_cells
        
        # Create response fields using Gaussian filters
        for i in range(self.n_cells):
            cells[i, place_centers[i][0], place_centers[i][1]] = 1
            cells[i] = gaussian_filter(cells[i], sigma_list[i])
            if self.dg_ratio > 1:  # Create inhibitory surround if requested
                cells[i] -= gaussian_filter(cells[i], sigma_list[i]*self.dg_ratio)

        # Normalize responses if requested
        if self.normalize:
            if self.dg_ratio > 1:
                cells = cells * self.magnitude if self.magnitude is not None else cells
                cells = (cells - cells.mean(axis=(1, 2), keepdims=True))
            else:
                # Normalize to 0-1
                cells = (cells - cells.min()) / (cells.max() - cells.min())
                cells = cells * self.magnitude if self.magnitude is not None else cells
        self.response_map = cells

    def get_specs(self):
        """
        Get statistical specifications of the place cell population.
        
        Returns:
            dict: Dictionary containing population statistics including:
                - cell_max_avg: Average maximum firing rate across cells
                - cell_min_avg: Average minimum firing rate across cells  
                - cell_mean_avg: Average mean firing rate across cells
        """
        specs = super().get_specs()
        specs['cell_max_avg'] = self.response_map.max(axis=(1, 2)).mean()
        specs['cell_min_avg'] = self.response_map.min(axis=(1, 2)).mean()
        specs['cell_mean_avg'] = self.response_map.mean(axis=(1, 2)).mean()
        return specs
