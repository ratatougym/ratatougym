import numpy as np
from .sm_base import SMBase
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pickle


class GridCell(SMBase):
    """
    Grid cell responses for a spatially modulated grid-like field.
    """
    sens_type = 'grid_cell'
    def __init__(self, arena, **kwargs):
        """Initialize the grid cell responses.
        
        Args:
            arena: The arena object defining the spatial environment.
            **kwargs: Additional keyword arguments including:
                n_cells (int): Number of cells.
                sigma2scale (float): Scale to sigma ratio.
                scale (int or float): Scale (periodicity) of the grid cells.
                magnitude (float): Maximum magnitude of the cell responses.
                normalize (bool): If True, normalize the responses.
                orientation (float): Orientation of the grid cells (degrees).
        """
        super().__init__(arena, **kwargs)

        # Parameters
        self.scale = kwargs.get('scale', 40)
        self.sigma2scale = kwargs.get('sigma2scale', 3.26)
        self.magnitude = kwargs.get('magnitude', None)
        self.normalize = kwargs.get('normalize', False)
        self.orientation = kwargs.get('orientation', 0)
        self.orientation = int(self.orientation)

        # Validate and initialize parameters
        self._check_params()
        self._init_params()
        self._init_response_map()

    def _check_params(self):
        """Validate the parameters."""
        assert self.n_cells > 0, "n_cells must be greater than 0."
        if isinstance(self.sigma2scale, (int, float)):
            assert self.sigma2scale > 1, "sigma2scale must be greater than 1."
        elif isinstance(self.sigma, list):
            assert all(s > 0 for s in self.sigma), "All sigma values must be greater than 0."

    def _init_params(self):
        """Initialize and scale parameters based on spatial resolution."""
        if isinstance(self.scale, (int, float)):
            self.scale = self.scale / self.arena.spatial_resolution
        elif isinstance(self.sigma, list):
            self.scale = np.array(self.sigma) / self.arena.spatial_resolution
        else:
            raise ValueError("scale or sigma must be a float or a list.")
        self.sigma = self.scale / self.sigma2scale / 2

    def _generate_grid_phase_shifts(self):
        """Generate uniformly distributed points within a triangular region.
        
        Returns:
            points (ndarray): Array of shape (n_cells, 2) with uniformly distributed points.
        """
        # Generate random points
        u, v = self.rng.uniform(size=self.n_cells), self.rng.uniform(size=self.n_cells)

        # Reflect points to ensure uniformity in the triangular region
        mask = u + v > 1
        u[mask], v[mask] = 1 - u[mask], 1 - v[mask]

        # Define vertices of the triangle
        s = self.scale
        A, B, C = (0, 0), (0, -s), (np.sqrt(3) / 2 * s, -s / 2)

        # Convert to Cartesian coordinates
        self.grid_shifts = (1 - u - v)[:, None] * np.array(A) + \
                        u[:, None] * np.array(B) + \
                        v[:, None] * np.array(C)

    def _generate_grid_centers(self, extended_arena_dims):
        """Generate grid centers for the extended arena dimensions with optional rotation.

        Args:
            extended_arena_dims (tuple): Dimensions of the extended arena.

        Returns:
            grid_centers (ndarray): Array of grid center coordinates.
        """
        # Calculate grid dimensions and add padding for phase shift
        arc_length = extended_arena_dims[1] * 2 / np.sqrt(3)
        dim_0 = extended_arena_dims[0] + arc_length / 2
        dim_1 = arc_length
        dim_0 += self.sigma * 4
        dim_1 += self.sigma * 4

        # Create ranges for grid centers
        x, y = np.arange(0, dim_0, self.scale), np.arange(0, dim_1, self.scale)
        XX, YY = np.meshgrid(x, y)
        grid_centers = np.column_stack((XX.ravel(), YY.ravel()))

        # Apply hexagonal transformation
        trans_mat = np.array([[1, 1/2], [0, np.sqrt(3)/2]])
        grid_centers = grid_centers @ trans_mat.T
        grid_centers[:, 0] -= arc_length / 2  # Center the grid

        # Rotate the grid centers around the center of the dimensions
        if self.orientation != 0:
            # Convert rotation angle to radians
            angle_rad = np.radians(self.orientation)

            # Define rotation matrix
            rotation_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad),  np.cos(angle_rad)]
            ])

            # Compute the center of the dimensions
            center = np.array(extended_arena_dims) / 2

            # Translate grid centers to origin, apply rotation, and translate back
            grid_centers = (grid_centers - center) @ rotation_matrix.T + center

        return np.int32(grid_centers)

    def _generate_grid_map(self):
        """
        Generate standard grid centers for all cells without shifts.
        """
        diag = np.linalg.norm(self.arena.dimensions)
        max_shift = np.max(np.abs(self.grid_shifts))
        extended_dims = (int(diag * 2 + max_shift * 2), int(diag * 2 + max_shift * 2))  # Extended arena size
        grid_centers = self._generate_grid_centers(extended_dims)

        # Filter grid centers within bounds
        is_within_bounds = (
            (grid_centers[:, 0] >= 0) & (grid_centers[:, 0] < extended_dims[0]) &
            (grid_centers[:, 1] >= 0) & (grid_centers[:, 1] < extended_dims[1])
        )
        grid_centers = grid_centers[is_within_bounds]

        # Create sample map and apply Gaussian filter
        sample_map = np.zeros(extended_dims)
        sample_map[grid_centers[:, 0], grid_centers[:, 1]] = 1
        self.sample_map = gaussian_filter(sample_map, self.sigma, mode='constant')

    def _compute_res(self):
        """
        Compute the responses for the grid cells.
        """
        # Compute a sample map
        self._generate_grid_map()

        # Generate responses for each cell
        dims = self.arena.dimensions
        x_start = (self.sample_map.shape[0] - dims[0]) // 2
        y_start = (self.sample_map.shape[1] - dims[1]) // 2

        cells = np.zeros((self.n_cells, *dims))
        for cell_idx in range(self.n_cells):
            _x = int(x_start + self.grid_shifts[cell_idx, 0])
            _y = int(y_start + self.grid_shifts[cell_idx, 1])
            cell = self.sample_map[_x:_x + dims[0], _y:_y + dims[1]]
            if self.normalize:
                cell = (cell - cell.min()) / (cell.max() - cell.min())
            cells[cell_idx] = cell

        # Scale cells to have desired mean magnitude instead of max magnitude
        if self.magnitude is not None:
            # Calculate current mean for each cell
            cell_means = cells.mean(axis=(1, 2))
            # Create scaling factors to achieve target mean magnitude
            scaling_factors = self.magnitude / cell_means
            # Apply scaling factors to each cell (broadcasting across spatial dimensions)
            cells = cells * scaling_factors[:, np.newaxis, np.newaxis]
        
        self.response_map = cells

    def _init_response_map(self):
        """
        Initialize spatially modulated responses for the grid cells.
        """
        assert len(self.arena.dimensions) == 2, "Only 2D arenas are supported."
        super()._init_response_map()
        self._generate_grid_phase_shifts()
        self._compute_res()

    def get_specs(self):
        """Retrieve statistics of the cell responses."""
        specs = super().get_specs()
        specs.update({
            'cell_max_avg': self.response_map.max(axis=(1, 2)).mean(),
            'cell_min_avg': self.response_map.min(axis=(1, 2)).mean(),
            'cell_mean_avg': self.response_map.mean(axis=(1, 2)).mean(),
        })
        return specs

    def state_dict(self):
        """Get the essential attributes of the GridCell object.

        Returns:
            dict: Dictionary of the essential attributes.
        """
        return {
            'sens_type': self.sens_type,
            'n_cells': self.n_cells,
            'sigma': self.sigma,
            'scale': self.scale,
            'magnitude': self.magnitude,
            'normalize': self.normalize,
            'orientation': self.orientation,
            'grid_shifts': self.grid_shifts,
        }

    @classmethod
    def load_from_dict(cls, state_dict, arena):
        """Load the GridCell object from a dictionary and reconstruct it.

        Args:
            state_dict (dict): Dictionary containing the object's state.
            arena (Arena): Arena object to reinitialize the class.

        Returns:
            GridCell: Reconstructed GridCell object.
        """
        obj = cls(
            arena,
            n_cells=state_dict['n_cells'],
            sigma=state_dict['sigma'],
            scale=state_dict['scale'],
            magnitude=state_dict['magnitude'],
            normalize=state_dict['normalize'],
            orientation=state_dict['orientation']
        )
        obj.grid_shifts = state_dict['grid_shifts']
        obj._compute_res()

        return obj
