import torch
import numpy as np
import warnings

from rtgym.dataclass import Trajectory

warnings.filterwarnings('ignore', message='invalid value encountered in divide')


def combine_trajectories(traj_list: list):
    """
    Merge a list of Trajectory objects into a single Trajectory object
    """
    n_trajs = len(traj_list)
    coords_float, head_directions, displacements = [], [], []
    for i in range(n_trajs):
        traj = traj_list[i]
        coords_float.append(traj.float_coords)
        head_directions.append(traj.hds)
        displacements.append(traj.disps)
    return Trajectory(
        coords_float=np.concatenate(coords_float, axis=0),
        head_directions=np.concatenate(head_directions, axis=0),
        displacements=np.concatenate(displacements, axis=0),
    )


def states_to_ratemap(states, coords, arena_map):
    """
    Takes the states and coords and returns the firing fields using PyTorch.
    
    Args:
        states: shape=(n_batches, n_timesteps, n_cells) or (n_timesteps, n_batches, n_cells)
        coords: shape=(n_batches, n_timesteps, 2) or (n_timesteps, n_batches, 2)
        arena_map: shape=(n_x, n_y), 0 for free space, 1 for walls (torch.Tensor)

    Returns:
        field: shape=(n_cells, n_x, n_y) (torch.Tensor)
    """
    # Convert to PyTorch tensors if numpy arrays
    if isinstance(states, np.ndarray):
        states = torch.from_numpy(states)
    if isinstance(coords, np.ndarray):
        coords = torch.from_numpy(coords)
    
    # Ensure tensors are in continuous memory layout
    if not states.is_contiguous():
        states = states.contiguous()
    if not coords.is_contiguous():
        coords = coords.contiguous()

    # Ensure tensors have consistent data types
    states = states.float()
    coords = torch.round(coords).long().to(states.device)

    # Reshape states and coords
    n_batches, n_timesteps, n_cells = states.shape
    coords = coords.view(-1, 2)  # (n_batches * n_timesteps, 2)
    states = states.view(-1, n_cells)  # (n_batches * n_timesteps, n_cells)

    # Get gym background dimensions
    dims = arena_map.shape

    # Initialize firing fields and visit counts
    firing_fields = torch.zeros((n_cells, *dims), dtype=torch.float32, device=states.device)
    visit_counts = torch.zeros(dims, dtype=torch.float32, device=states.device)

    # Flatten coords for linear indexing
    flat_coords = coords[:, 0] * dims[1] + coords[:, 1]  # Linear indices for coords
    flat_fields = firing_fields.view(n_cells, -1)
    flat_counts = visit_counts.view(-1)

    # Accumulate firing fields and visit counts
    flat_fields.index_add_(1, flat_coords, states.T)
    flat_counts.index_add_(0, flat_coords, torch.ones_like(flat_coords, dtype=torch.float32))

    # Normalize firing fields by visit counts
    firing_fields /= visit_counts.clamp(min=1).unsqueeze(0)  # Avoid division by zero
    
    # Set unvisited areas to NaN
    firing_fields[:, visit_counts == 0] = float('nan')

    return firing_fields


def get_gym_dimensions(coords, arena_map):
    """
    Get the dimensions of the gym background
    """
    if arena_map is None:
        # Infer dimensions from coords
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        return x_max - x_min + 1, y_max - y_min + 1
    else:
        return arena_map.shape


def states2ff(states, coords, arena_map):
    """
    Takes the states and coords and returns the firing fields.
    
    Args:
        states: shape=(n_batches, n_timesteps, n_cells) or (n_timesteps, n_batches, n_cells)
        coords: shape=(n_batches, n_timesteps, 2) or (n_timesteps, n_batches, 2)
        gym_bg: shape=(n_x, n_y), 0 for free space, 1 for walls

    Returns:
        field: shape=(n_cells, n_x, n_y)
    """
    assert states.shape[0] == coords.shape[0]
    assert states.shape[1] == coords.shape[1]

    # reshape coords and states
    coords = coords.reshape(coords.shape[0] * coords.shape[1], coords.shape[2])
    states = states.reshape(states.shape[0] * states.shape[1], states.shape[2])

    if isinstance(states, torch.Tensor):
        states = states.cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.cpu().numpy()

    return restrain2ff(coords, states, arena_map)


def restrain2ff(coords, states, arena_map):
    n_cells = states.shape[1]
    dimensions = arena_map.shape
    dim = len(dimensions)
    coords = np.round(coords).astype(np.int32)
    if dim == 2:
        field = np.full((n_cells, dimensions[0], dimensions[1]), 0, dtype=np.float64)
        counts = np.zeros((dimensions[0], dimensions[1]), dtype=np.int32)
        for i in range(coords.shape[0]):
            x, y = coords[i]
            field[:, x, y] += states[i]
            counts[x, y] += 1
        field = np.where(counts > 0, field / counts, np.nan)
    elif dim == 3:
        field = np.full((n_cells, dimensions[0], dimensions[1], dimensions[2]), 0, dtype=np.float64)
        counts = np.zeros((dimensions[0], dimensions[1], dimensions[2]), dtype=np.int32)
        for i in range(coords.shape[0]):
            x, y, z = coords[i]
            field[:, x, y, z] += states[i]
            counts[x, y, z] += 1
        for i in range(n_cells):
            field[i] = np.where(counts > 0, field[i] / counts, np.nan)
    return field


class RatemapAggregator:
    def __init__(self, arena_map, device=None):
        """
        Class to accumulate partial data for rate-map computation.

        Args:
            arena_map (torch.Tensor or np.ndarray):
                Map of shape (n_x, n_y), 0 for free space, 1 for walls, 
                used for figuring out dimensions and for masking.
            device (str or torch.device, optional):
                The device on which to store the data (CPU or GPU). 
                If None, uses arena_map device if it's a torch.Tensor, 
                otherwise "cpu".
        """
        # If arena_map is numpy array, convert to torch
        if isinstance(arena_map, np.ndarray):
            arena_map = torch.from_numpy(arena_map)
        
        self.arena_map = arena_map
        self.dims = arena_map.shape  # (n_x, n_y)
        self.n_cells = None
        # Infer device
        if device is None:
            self.device = arena_map.device if arena_map.is_cuda else torch.device('cpu')
        else:
            self.device = torch.device(device)

    
    def init_counts(self):
        self.partial_sums = torch.zeros(
            (self.n_cells, *self.dims),
            dtype=torch.float32,
            device=self.device
        )
        # shape: (n_x, n_y)
        self.visit_counts = torch.zeros(
            self.dims,
            dtype=torch.float32,
            device=self.device
        )

    def update(self, states, coords):
        """
        Accumulate partial sums and visit counts from new data.

        Args:
            states (torch.Tensor or np.ndarray):
                Shape=(n_batches, n_timesteps, n_cells) or (n_timesteps, n_batches, n_cells).
            coords (torch.Tensor or np.ndarray):
                Shape=(n_batches, n_timesteps, 2) or (n_timesteps, n_batches, 2).
        """
        if self.n_cells is None:
            self.n_cells = states.shape[-1]
            self.init_counts()

        # Convert to torch if numpy
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)

        # Move to the same device
        states = states.to(self.device)
        coords = coords.to(self.device)

        # Ensure correct dtype
        states = states.float()
        coords = torch.round(coords).long()  # round and convert to long

        # Standardize shapes
        assert states.dim() == 3, "states must have 3 dims: (n_batches, n_timesteps, n_cells)"
        assert coords.dim() == 3, "coords must have 3 dims: (n_batches, n_timesteps, 2)"

        # Flatten
        coords = coords.view(-1, 2)     # (n_batches * n_timesteps, 2)
        states = states.view(-1, self.n_cells)  # (n_batches * n_timesteps, n_cells)

        # Flatten partial sums and visit_counts for fast index_add
        flat_sums = self.partial_sums.view(self.n_cells, -1)  # shape: (n_cells, n_x*n_y)
        flat_counts = self.visit_counts.view(-1)              # shape: (n_x*n_y)

        # Convert (row, col) coords into linear indices
        dims = self.dims
        flat_coords = coords[:, 0] * dims[1] + coords[:, 1]  # shape: (n_samples,)

        # Accumulate partial sums
        # states.T shape: (n_cells, n_samples)
        # so we add states.T to the flat_sums at the flattened coordinate indices
        flat_sums.index_add_(1, flat_coords, states.T)

        # Accumulate visit counts
        flat_counts.index_add_(
            0, 
            flat_coords, 
            torch.ones_like(flat_coords, dtype=torch.float32)
        )

    def get_ratemap(self):
        """
        Returns the final normalized firing fields (n_cells, n_x, n_y).
        Unvisited points (visit_count=0) will be NaN.
        """
        # Avoid division by zero by clamping
        denom = self.visit_counts.clamp(min=1.0)  # shape: (n_x, n_y)

        # temp_counts = self.visit_counts[5:-5, 5:-5]
        # print(temp_counts.min(), temp_counts.max(), temp_counts.mean(), temp_counts.std())
        
        # Broadcasting: partial_sums shape (n_cells, n_x, n_y) / (n_x, n_y)
        ratemap = self.partial_sums / denom.unsqueeze(0)

        # Set unvisited areas to NaN
        mask_unvisited = (self.visit_counts == 0)
        ratemap[:, mask_unvisited] = float('nan')

        return ratemap

    def reset(self):
        """
        Reset the aggregator (clears all partial sums and counts).
        """
        self.partial_sums.zero_()
        self.visit_counts.zero_()
