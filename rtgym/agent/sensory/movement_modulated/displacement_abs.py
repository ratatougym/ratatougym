import numpy as np

import rtgym.dataclass
from .mm_base import MMBase
import rtgym


class DisplacementAbs(MMBase):
    """Absolute displacement signal sensory cells.
    
    Generates sensory responses based on the absolute displacement of the agent.
    Displacement values may include negative values and represent spatial movement
    in both x and y directions.
    
    Args:
        arena (Arena): Arena environment object.
        **kwargs: Additional keyword arguments including:
            n_cells (int): Number of displacement cells (must be even).
            sigma_s (float): Temporal smoothing sigma in seconds.
            ssigma_s (float): Spatial smoothing sigma in seconds.
            
    Attributes:
        sens_type (str): Sensory type identifier 'displacement_abs'.
    """
    
    sens_type = 'displacement_abs'
    
    def __init__(self, arena, **kwargs):
        super().__init__(arena, **kwargs)

        # parameters
        self.sigma_ts = self._t_to_ts(kwargs.get('sigma_s', 0.0))
        self.ssigma_ts = self._t_to_ts(kwargs.get('ssigma_s', 0.0))

        # check parameters
        self._check_params()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"
        assert self.n_cells % 2 == 0, "n_cells must be even"

    def get_response(self, agent_data: rtgym.dataclass.Trajectory):
        assert isinstance(agent_data, rtgym.dataclass.Trajectory), \
            "agent_data for displacement_abs must be a rtgym.dataclass.Trajectory"
        dup_res = self._duplicate_res(agent_data.disp, divisor=2)
        return self._smooth_res(dup_res)

    def get_specs(self):
        specs = super().get_specs()
        return specs
