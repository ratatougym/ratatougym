import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from rtgym.dataclass import AgentState, Trajectory
from .displacement_abs import DisplacementAbs
from typing import Any


class SpeedCell(DisplacementAbs):
    """Speed cells that encode movement speed.
    
    Generates sensory responses based on the absolute speed (velocity magnitude) of agent movement.
    Each cell outputs a 1D value representing the agent's speed, regardless of direction.
    Inherits from DisplacementAbs and converts displacement to speed.
    
    Args:
        arena (Arena): Arena environment object.
        **kwargs: Additional keyword arguments including:
            n_cells (int): Number of speed cells.
            sigma_s (float): Temporal smoothing sigma in seconds.
            ssigma_s (float): Spatial smoothing sigma in seconds.
            normalize (bool): Whether to normalize cell responses.
            
    Attributes:
        sens_type (str): Sensory type identifier 'speed_cell'.
    """
    
    sens_type = 'speed_cell'

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

    @staticmethod
    def _displacement_to_velocity(displacement):
        """Convert displacement to speed.
        
        Computes the Euclidean norm of 2D displacement vectors to get
        speed (velocity magnitude).
        
        Args:
            displacement (np.ndarray): Displacement array of shape (n_batch, n_time, 2).
            
        Returns:
            np.ndarray: Speed values of shape (n_batch, n_time, 1).
        """
        # displacement: (n_batch, n_time, 2), last dimension is x, y
        # speed: (n_batch, n_time, 1), the magnitude of velocity
        velocity = np.linalg.norm(displacement, axis=2)
        velocity = velocity[:, :, np.newaxis]
        return velocity

    def get_response(self, agent_data: Union[AgentState, Trajectory, Any]):
        """Get speed responses for given agent data.
        
        Converts displacement to speed and duplicates across all cells.
        Each cell outputs the same speed value but can be modified by smoothing.
        
        Args:
            agent_data (AgentState or Trajectory): Agent data containing displacement.
            
        Returns:
            np.ndarray: Speed responses.
            
        Raises:
            ValueError: If agent_data type is not supported.
        """
        if isinstance(agent_data, Trajectory):
            vel = self._displacement_to_velocity(agent_data.disp) / self.t_res
            dup_res = self._duplicate_res(vel)
            return self._smooth_res(dup_res)
        elif isinstance(agent_data, AgentState):
            return self._displacement_to_velocity(agent_data.disp) / self.t_res
        else:
            raise ValueError(f"Invalid agent_data type: {type(agent_data)}, must be rtgym.dataclass.Trajectory or rtgym.dataclass.AgentState")

    def get_specs(self):
        """Get specifications of this sensory modality.
        
        Returns:
            dict: Dictionary containing speed cell specifications.
        """
        specs = super().get_specs()
        return specs

    def vis(self, N=5, max_vel=1.0, n_bins=10, cmap='jet', *args, **kwargs):
        """Visualize speed response curves.
        
        Plots speed vs response for the first N cells.
        
        Args:
            N (int): Number of cells to visualize.
            max_vel (float): Maximum speed for visualization range.
            n_bins (int): Number of speed bins to plot.
            cmap (str): Colormap for visualization.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        vels = np.linspace(0, max_vel, n_bins).reshape(1, -1, 1)
        dups_res = self._duplicate_res(vels)
        smooth_res = self._smooth_res(dups_res)
        
        fig, axs = plt.subplots(1, N, figsize=(4*N, 4))
        for i in range(N):
            # plot, x axis is the speed, y axis is the response
            v = vels[0, :, 0]
            r = smooth_res[0, :, i]
            axs[i].plot(v, r)
        plt.show()
