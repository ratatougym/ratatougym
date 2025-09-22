import numpy as np
from .sm_base import SMBase
from scipy.signal import fftconvolve


class BoundaryCell(SMBase):
    """Boundary cells that respond based on distance to walls.
    
    These cells fire based on the distance to the nearest boundary in any direction.
    Note: This implementation may not be biologically realistic as it responds
    to boundaries in all directions simultaneously.
    
    Args:
        arena (Arena): Arena environment object.
        **kwargs: Additional keyword arguments including:
            n_cells (int): Number of boundary cells.
            res_dist (float): Response distance from boundary in spatial units. Defaults to 10.
            magnitude (float): Maximum magnitude of cell responses.
            normalize (bool): Whether to normalize cell responses.
            center_border_ratio (float): Ratio of center to border cells (0-1). Defaults to 0.5.
            
    Attributes:
        sens_type (str): Sensory type identifier 'boundary_cell'.
    """
    
    sens_type = 'boundary_cell'
    
    def __init__(self, arena, **kwargs):
        super(BoundaryCell, self).__init__(arena, **kwargs)
        
        # parameters
        self.res_dist = kwargs.get('res_dist', 10) / self.arena.spatial_resolution
        self.magnitude = kwargs.get('magnitude', None)
        self.normalize = kwargs.get('normalize', False)
        self.center_border_ratio = kwargs.get('center_border_ratio', 0.5)
        
        # check parameters and initialize responses
        self._check_params()
        self._init_response_map()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"
        # check res_dist
        assert self.res_dist > 0, "res_dist <= 0"
        assert self.res_dist < self.arena.dimensions[1], "res_dist >= arena.dimensions[1]"
        assert self.res_dist < self.arena.dimensions[0], "res_dist >= arena.dimensions[0]"
    
    def _init_response_map(self):
        """ Initialize response_map """
        super(BoundaryCell, self)._init_response_map()
        
        # initialize response_map
        cell_res_dist = self.rng.normal(self.res_dist, self.res_dist/4, self.n_cells).astype(int)
        cell_res_dist = np.clip(cell_res_dist, 0, None)
        for i in range(self.n_cells):
            kernal = np.ones((cell_res_dist[i]*2, cell_res_dist[i]*2))
            self.response_map[i, :, :] = -fftconvolve(self.arena.inv_arena_map, kernal, mode='same')/kernal.sum()
            self.response_map[i, :, :] *= self.rng.choice([-1, 1], 1, p=[self.center_border_ratio, 1-self.center_border_ratio])
            self.response_map[i, :, :] -= self.response_map[i, :, :].min()
        
        # normalize response_map
        if self.normalize:
            self.response_map = (self.response_map - self.response_map.min()) / (self.response_map.max() - self.response_map.min())
        
        # set magnitude
        if self.magnitude is not None:
            self.response_map = self.magnitude * self.response_map

    def get_specs(self):
        specs = super().get_specs()
        specs['cell_max_avg'] = self.response_map.max(axis=(1, 2)).mean()
        specs['cell_min_avg'] = self.response_map.min(axis=(1, 2)).mean()
        specs['cell_mean_avg'] = self.response_map.mean(axis=(1, 2)).mean()
        return specs
