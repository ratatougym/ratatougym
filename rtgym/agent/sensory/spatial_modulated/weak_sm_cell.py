import numpy as np
from .sm_base import SMBase
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


class WeakSMCell(SMBase):
    """Weakly spatially modulated cells with Gaussian responses.
    
    These cells have broader, weaker spatial tuning compared to place cells,
    representing cells that are spatially modulated but not strongly place-specific.
    Each cell has a Gaussian-shaped firing field at a random location.
    
    Args:
        arena (Arena): Arena environment object.
        **kwargs: Additional keyword arguments including:
            n_cells (int): Number of weakly spatially modulated cells.
            sigma (float): Sigma of the Gaussian filter in spatial units.
            magnitude (float): Maximum magnitude of cell responses.
            
    Attributes:
        sens_type (str): Sensory type identifier 'weak_sm_cell'.
        response_map (np.ndarray): Spatial response map of shape (n_cells, *arena_dimensions).
    """
    
    sens_type = 'weak_sm_cell'
    
    def __init__(self, arena, **kwargs):
        super().__init__(arena, **kwargs)
        
        # parameters
        self.sigma = kwargs.get('sigma', 8)
        self.ssigma = kwargs.get('ssigma', 0)
        self.magnitude = kwargs.get('magnitude', None)
        self.normalize = kwargs.get('normalize', False)
        
        # check parameters and initialize responses
        self._check_params()
        self._init_params()
        self._init_response_map()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"
        # sigma can be a list or an integer
        if isinstance(self.sigma, int):
            assert self.sigma > 0, "sigma <= 0"
        elif isinstance(self.sigma, list):
            for s in self.sigma:
                assert s > 0, "sigma <= 0"

    def _init_params(self):
        if isinstance(self.sigma, int):
            self.sigma = int(self.sigma/self.arena.spatial_resolution)
        elif isinstance(self.sigma, list):
            self.sigma = [int(s/self.arena.spatial_resolution) for s in self.sigma]

    def _init_response_map(self):
        """ Initialize response_map """
        super()._init_response_map()
        # border padding is also included in the arena map
        self.response_map = self._generate_smcells()

    def _generate_smcells(self):
        """ 
        Generate a spatially modulated non-grid/place cell response field 
        """
        arena_dims = self.arena.dimensions
        cells = self.rng.normal(0, 1, (self.n_cells, *arena_dims))
        # filter each cell response field with a 2d gaussian filter
        if self.ssigma > 0:
            mean, std = self.sigma, self.ssigma
            a = (0 - mean) / std
            b = (100 - mean) / std
            sigma_distribution = truncnorm(a=a, b=b, loc=mean, scale=std)
            sigma_list = sigma_distribution.rvs(self.n_cells, random_state=self.rng)
        else:
            sigma_list = np.ones(self.n_cells) * self.sigma
        for i in range(self.n_cells):
            cell = gaussian_filter(cells[i], sigma_list[i], mode='constant')
            if self.normalize:
                cell = (cell - cell.min()) / (cell.max() - cell.min())
            cells[i] = cell
        
        # Scale cells to have desired mean magnitude instead of max magnitude
        if self.magnitude is not None:
            # Calculate current mean for each cell
            cell_means = cells.mean(axis=(1, 2))
            # Create scaling factors to achieve target mean magnitude
            scaling_factors = self.magnitude / cell_means
            # Apply scaling factors to each cell (broadcasting across spatial dimensions)
            cells = cells * scaling_factors[:, np.newaxis, np.newaxis]
        
        return cells # (n, *arena_dims)

    def get_specs(self):
        specs = super().get_specs()
        specs['cell_max_avg'] = self.response_map.max(axis=(1, 2)).mean()
        specs['cell_min_avg'] = self.response_map.min(axis=(1, 2)).mean()
        specs['cell_mean_avg'] = self.response_map.mean(axis=(1, 2)).mean()
        return specs
