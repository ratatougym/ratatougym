import numpy as np
import rtgym
import rtgym.dataclass
from .mm_base import MMBase
import matplotlib.pyplot as plt
from rtgym.dataclass import Trajectory, AgentState
from typing import Union

class DirectionCell(MMBase):
    """Direction cells with Gaussian tuning to movement direction.
    
    These cells respond to the direction of agent movement with Gaussian tuning
    curves. Each cell has a preferred direction and fires maximally when the
    agent moves in that direction.
    
    Args:
        arena (Arena): Arena environment object.
        **kwargs: Additional keyword arguments including:
            n_cells (int): Number of direction cells.
            magnitude (float): Maximum magnitude of cell responses.
            normalize (bool): Whether to normalize cell responses.
            sorted (bool): Whether to sort cells by preferred directions. Defaults to True.
            n_bins (int): Number of direction bins for discretization. Defaults to 360.
            sigma (float): Standard deviation of the Gaussian tuning kernel. Defaults to 2.
            msigma (float): Standard deviation of magnitude modulation.
            
    Attributes:
        neuron_type (str): Sensory type identifier 'direction_cell'.
    """
    
    neuron_type = 'direction_cell'
    
    def __init__(self, arena, **kwargs):
        super().__init__(arena, **kwargs)

        # parameters
        self.sigma = kwargs.get('sigma', 2)
        self.n_bins = kwargs.get('n_bins', 360)
        self.sorted = kwargs.get('sorted', True)
        self.normalize = kwargs.get('normalize', False)

        # check parameters
        self._check_params()
        self._init_mm_responses()

    def _check_params(self):
        """ Check parameters """
        assert self.n_cells > 0, "n_cells <= 0"
        assert self.n_bins > 0, "n_bins <= 0"

    def _init_mm_responses(self):
        """
        Initialize the movement modulated responses with wrapped Gaussian distributions.
        """
        # Define the range of directions
        self.dirs = np.linspace(-np.pi, np.pi, self.n_bins, endpoint=False)  # Shape: (n_bins,)

        # Assign random preferred directions to each cell
        self.pref_dirs = self.rng.uniform(-np.pi, np.pi, self.n_cells)  # Shape: (n_cells,)

        if self.sorted:
            self.pref_dirs = np.sort(self.pref_dirs)

        # Compute the angular distance between each cell's preferred direction and each bin's direction (n_cells, n_bins)
        angular_dist = np.abs(self.pref_dirs[:, np.newaxis] - self.dirs[np.newaxis, :])
        angular_dist = np.mod(angular_dist + np.pi, 2 * np.pi) - np.pi

        # Compute the Gaussian response based on angular distance
        self.mm_responses = np.exp(-0.5 * (angular_dist / self.sigma) ** 2)  # Shape: (n_cells, n_bins)

        if self.normalize:
            # zero-one normalization
            self.mm_responses = (self.mm_responses - np.min(self.mm_responses)) / (np.max(self.mm_responses) - np.min(self.mm_responses))

    def get_response(self, agent_state: Union[AgentState, Trajectory]):
        disp = agent_state.disp
        # arctan2 already returns angles in [-π, π]
        mv_dir = np.arctan2(disp[:, :, 0], disp[:, :, 1])
        
        # Squeeze if the last dimension is singleton
        if mv_dir.ndim == 3 and mv_dir.shape[-1] == 1:
            mv_dir = mv_dir.squeeze(-1)

        # Directly compute bin indices from angles
        indices = (((mv_dir + np.pi) / (2 * np.pi)) * self.n_bins).astype(np.int64)
        
        # Use np.take to get responses from precomputed mm_responses
        responses = np.take(self.mm_responses, indices, axis=1)
        
        return responses.transpose(1, 2, 0) * self.magnitude

    def get_specs(self):
        specs = super().get_specs()
        specs['magnitude'] = self.magnitude
        specs['normalize'] = self.normalize
        return specs

    def vis(self, N, **kwargs):
        fig, ax = plt.subplots()
        cax = ax.imshow(self.mm_responses[:N], aspect='auto', cmap='jet')
        fig.colorbar(cax, ax=ax)
        ax.set_xticks([0, self.n_bins//4, self.n_bins//2, 3*self.n_bins//4, self.n_bins-1])
        ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π (wrap around)'])
        ax.set_xlabel('Direction (radian)')
        ax.set_ylabel('Cells')
        ax.set_title('Head Direction Cells')
        plt.show()
