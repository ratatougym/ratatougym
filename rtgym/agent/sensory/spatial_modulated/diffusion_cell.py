"""Spatial random fields with wall-aware diffusion from grid_and_place."""

import torch
import torch.nn.functional as F
from rtgym.utils.common import hash_seed
from .sm_base import SMBase


class DiffusionCell(SMBase):
    """Smooth random fields in 2D or 3D without averaging through wall cells.

    This preserves grid_and_place's box-kernel diffusion and normalization.
    It is a separate sensory type because it differs from Gaussian smoothing.
    """

    sens_type = 'diffusion_cell'

    def __init__(self, arena, n_cells, sensory_key, sigma=8, magnitude=None,
                 normalize=False, seed=None, device='cpu', **kwargs):
        super().__init__(arena, n_cells, sensory_key, seed=seed)
        if sigma <= 0:
            raise ValueError('sigma must be positive.')
        self.sigma = sigma / arena.spatial_resolution
        self.magnitude = magnitude
        self.normalize = normalize
        self.device = torch.device(device)

        # Seed field generation independently of the control RNG.
        self.field_rng = torch.Generator(device='cpu')
        if seed is not None:
            field_seed = hash_seed(seed, sensory_key)
            self.field_rng.manual_seed(field_seed)
        self._init_response_map()

    def _init_response_map(self):
        # Draw on CPU to retain the original per-group seeded random fields.
        shape = (self.n_cells, *self.arena.dimensions)
        cells = torch.empty(shape)
        cells.normal_(0, 1, generator=self.field_rng)
        cells = cells.to(self.device)
        self.raw_field = cells.clone()
        cells = self.diffusion_smooth(cells)

        # Scale mean responses before masking walls, matching the source model.
        if self.magnitude is not None:
            axes = range(1, cells.ndim)
            axes = tuple(axes)
            means = cells.mean(dim=axes, keepdim=True)
            means = means.clamp(min=1e-8)
            scale = self.magnitude / means
            cells = cells * scale

        # Mask wall cells after matching the requested mean response.
        arena_map = self.arena.tensor_map(self.device)
        free_mask = arena_map == 0
        free_mask = free_mask.float()
        self.response_map = cells * free_mask
        self._tensor_response_map = None

    def diffusion_smooth(self, cells, free_mask=None):
        ndim = self.arena.ndim
        if free_mask is None:
            arena_map = self.arena.tensor_map(cells.device)
            free_mask = arena_map == 0
            free_mask = free_mask.float()

        # The repeated box kernel has variance approximately 2*n/3 per axis.
        iteration_count = round(1.5 * self.sigma * self.sigma)
        n_iters = max(1, iteration_count)
        kernel_shape = (1, 1) + (3,) * ndim
        kernel = torch.ones(kernel_shape, device=cells.device)
        kernel = kernel / 3**ndim
        padding = (1, 1) * ndim
        conv = F.conv2d if ndim == 2 else F.conv3d

        # Normalize each voxel by its free-neighbour weight, including the border.
        mask = free_mask[None, None]
        mask = F.pad(mask, padding, mode='replicate')
        neighbour_weight = conv(mask, kernel)
        neighbour_weight = neighbour_weight[0, 0]
        neighbour_weight = neighbour_weight.clamp(min=1e-8)

        # Diffuse only through the available free neighbours.
        cells = cells * free_mask
        for _ in range(n_iters):
            field = cells.unsqueeze(1)
            field = F.pad(field, padding, mode='replicate')
            smoothed = conv(field, kernel)
            smoothed = smoothed[:, 0]
            cells = (smoothed / neighbour_weight) * free_mask

        # Normalize over the entire grid, as in the original implementation.
        if self.normalize:
            axes = range(1, cells.ndim)
            axes = tuple(axes)
            minimum = cells.amin(dim=axes, keepdim=True)
            maximum = cells.amax(dim=axes, keepdim=True)
            span = maximum - minimum
            span = span.clamp(min=1e-8)
            cells = (cells - minimum) / span
        return cells
