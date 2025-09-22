import numpy as np

import rtgym.dataclass
from .displacement_abs import DisplacementAbs
import rtgym


class DirectionRad(DisplacementAbs):
    """Direction cells with raw radian output.
    
    Generates sensory responses based on the direction of agent movement,
    providing raw directional information in radians. Inherits from 
    DisplacementAbs and converts displacement to directional angles.
    
    Args:
        arena (Arena): Arena environment object.
        **kwargs: Additional keyword arguments including:
            n_cells (int): Number of direction cells.
            magnitude (float): Maximum magnitude of cell responses.
            normalize (bool): Whether to normalize cell responses.
            sigma_s (float): Temporal smoothing sigma in seconds. Defaults to 0.0.
            ssigma_s (float): Spatial smoothing sigma in seconds. Defaults to 0.0.
            
    Attributes:
        sens_type (str): Sensory type identifier 'direction_rad'.
    """
    
    sens_type = 'direction_rad'
    
    def __init__(self, arena, **kwargs):
        super().__init__(arena, **kwargs)

        # parameters
        self.magnitude = kwargs.get('magnitude', None)
        self.sigma_ts = self._t_to_ts(kwargs.get('sigma_s', 0.0))
        self.ssigma_ts = self._t_to_ts(kwargs.get('ssigma_s', 0.0))
        self.normalize = kwargs.get('normalize', False)

        # check parameters
        self._check_params()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"

    @staticmethod
    def _displacement_to_hd(displacement):
        """
        Convert displacement to direction
        """
        # displacement: (n_time, n_batch, 2), last dimension is x, y
        # direction: (n_time, n_batch, 1), the radian angle within [0, 2pi]
        dirs = np.arctan2(displacement[:, :, 0], displacement[:, :, 1])
        return dirs[:, :, np.newaxis]

    def get_responses(self, traj: rtgym.dataclass.Trajectory):
        disps = traj.disps  # [batch, timesteps, 2]
        dirs = self._displacement_to_hd(disps)
        dup_res = self._duplicate_res(dirs)
        return self._smooth_res(dup_res)

    def get_specs(self):
        specs = super().get_specs()
        specs['magnitude'] = self.magnitude
        specs['normalize'] = self.normalize
        return specs
