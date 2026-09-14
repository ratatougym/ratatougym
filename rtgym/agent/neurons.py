"""
The sensory module contains the Neurons class which is responsible for 
creating and managing spatially and movement-modulated sensory cells of the agent.
"""


import rtgym
import pickle
import numpy as np
import torch
from typing import Dict, Any, Union, List, Tuple

from rtgym.dataclasses import AgentState, Trajectory
from rtgym.utils.decode_response import (
    decode_response_euclidean,
    decode_response_kdtree,
    decode_response_torch,
    decode_response_faiss,
    decode_response_interpolation
)
from .sensory.spatial_modulated import *
from .sensory.spatial_modulated.sm_base import SMBase
from .sensory.movement_modulated import *


class Neurons:
    """
    The class object that manages the sensory system of the agent. More broadly, 
    it handles all simulated neuronal responses of the agent.  

    When the gym is initialized, an `Agent` object is automatically created, 
    which in turn creates a `Neurons` object. The `Neurons` object is initially 
    just a placeholder and must be initialized with a sensory profile (a dictionary) 
    that defines the simulated neuronal groups and their parameters.  

    During spatial traversal, RatatouGym separates the concerns of trajectory 
    generation and neuronal response computation. Once a trajectory is generated, 
    RatatouGym calls the `get_response` method of the `Neurons` object. This 
    method takes the trajectory as input and computes the corresponding neuronal 
    responses using the defined tuning curves.  

    This class should not be initialized directly. The `RatatouGym` class will 
    automatically manage it.

    Args:
        gym (RatatouGym): Parent RatatouGym object.
    """
    def __init__(self, gym):
        self.gym = gym
        self.neuron_groups = {}
        self.ranges = None  # Keep track of the indices of the sensory cells

    @property
    def t_res(self):
        return self.gym.t_res

    @property
    def s_res(self):
        return self.gym.s_res
    
    @property
    def arena(self):
        return self.gym.arena
    
    @property
    def common_params(self):
        return {'arena': self.arena, 't_res': self.t_res}

    def _on_arena_change(self):
        """
        Set the arena for the sensory cells.

        Args:
            arena: rtgym.arena.Arena object.
        """
        pass

    def list_all(self):
        """
        List all the sensory cells.
        """
        return list(self.neuron_groups)

    def init_from_profile(self, sensory_profile):
        # Initialize spatial and movement modulated cells
        self.neuron_groups = {}
        profile = sensory_profile if sensory_profile is not None else {}
        self._update_neurons(profile)
        self._update_ranges()
    
    def add_neuron_group(self, sensory_profile: Dict[str, Any]):
        """
        Add a sensory cell to the sensory system.

        Args:
            sensory_profile: Dictionary containing the sensory profile.
        """
        profile = sensory_profile if sensory_profile is not None else {}
        self._update_neurons(profile)
        self._update_ranges()

    def _update_neurons(self, profile_list):
        """
        Initializes the cells based on the provided profile list.
        """
        for key, value in profile_list.items():
            sensory_type = value.get('type', 'diffusion_cell')
            sensory_class = Neurons._get_sensory_class(sensory_type)
            if self.arena.ndim == 3 and sensory_class is not DiffusionCell:
                raise ValueError('Only diffusion_cell currently supports 3D arenas.')
            params = dict(value)
            if sensory_class is DiffusionCell:
                params.setdefault('device', self.gym.device)
            self.neuron_groups[key] = sensory_class(sensory_key=key, **self.common_params, **params)

    def _update_ranges(self):
        counts = [group.n_cells for group in self.neuron_groups.values()]
        _ranges = np.cumsum(counts)
        _ranges = np.insert(_ranges, 0, 0)
        _ranges = _ranges.tolist()
        self.ranges = {key: (_ranges[i], _ranges[i+1]) for i, key in enumerate(self.neuron_groups)}
    
    def filter_neurons(self, keys=None, str_filter=None, type_filter=None):
        """
        This helps to find the keys of the sensory cells that match the given criteria.

        It will prioritize the most specific filter. The specificity from most to least is:
            keys > str_filter > type_filter
        """
        if keys is not None:
            if isinstance(keys, str):
                return_keys = [keys]
            elif hasattr(keys, '__iter__'):
                return_keys = list(keys)
            else:
                raise ValueError(f"Unknown keys: {keys}")
        elif str_filter is not None:
            return_keys = [key for key in self.neuron_groups.keys() if str_filter in key]
        elif type_filter is not None:
            return_keys = [key for key, sensory_item in self.neuron_groups.items() if type_filter == sensory_item.sens_type]
        else:
            return_keys = list(self.neuron_groups)
        return sorted(return_keys)

    def num_neurons(self, keys=None, str_filter=None, type_filter=None):
        keys = self.filter_neurons(keys, str_filter, type_filter)
        return sum([self.neuron_groups[key].n_cells for key in keys])

    @staticmethod
    def _get_sensory_class(sensory_type):
        sensory_classes = {cls.sens_type: cls for cls in 
                           [WeakSMCell, PlaceCell, BoundaryCell, GridCell, DiffusionCell,
                            SpeedCell, DirectionCell, DirectionRad, DisplacementAbs, 
                            HeadDirectionCell]}

        assert sensory_type in sensory_classes, f"Unknown sensory type: {sensory_type}"
        return sensory_classes[sensory_type]
    
    def aggregate_res_maps(self, keys=None, str_filter=None, type_filter=None):
        """Aggregate sensory response maps from spatial modalities.
        
        Combines response maps from multiple spatial sensory modalities into
        a single array for analysis or decoding purposes.
        
        Args:
            keys (list, optional): Keys to filter the sensories.
            str_filter (str, optional): String filter for sensory names.
            type_filter (str, optional): Type filter for sensory modalities.
            
        Returns:
            np.ndarray: Aggregated sensory response maps of shape (n_cells, H, W).
            
        Raises:
            AssertionError: If non-spatial modulated sensory cells are included.
        """
        # Filter and aggregate the sensory response maps.
        keys = self.filter_neurons(keys, str_filter, type_filter)
        res_maps = []
        for key in keys:
            assert self.neuron_groups[key].sens_category == 'spatial_modulated', (
                "Only spatial_modulated sensory cells can be decoded into a trajectory"
            )
            response_map = self.neuron_groups[key].response_map
            if isinstance(response_map, torch.Tensor):
                response_map = response_map.detach()
                response_map = response_map.cpu()
                response_map = response_map.numpy()
            res_maps.append(response_map)
        
        # Concatenate along the cell dimension. Shape: (n_cells, H, W)
        return np.concatenate(res_maps, axis=0)

    def decode_response(self, response: np.ndarray, res_maps=None, keys=None, str_filter=None, type_filter=None,
                        use_torch=True, device=None, method="euclidean", **kwargs):
        """
        Decode sensory response into spatial coordinates using various optimization methods.

        This method converts high-dimensional sensory responses (e.g., from place cells,
        grid cells) back to spatial coordinates. Multiple algorithms are available,
        ranging from exact brute-force search to fast approximate methods.

        Args:
            response (np.ndarray): Neurons response array of shape:
                - (B, T, D) for trajectory decoding
                - (B, D) for single state decoding
                where B=batch size, T=time steps, D=feature dimensions
            res_maps (np.ndarray, optional): Precomputed response template maps.
                If None, computed from filtered sensory modalities.
            keys (list, optional): Specific sensory keys to include in decoding.
            str_filter (str, optional): String filter for sensory names.
            type_filter (str, optional): Type filter for sensory modalities.
            use_torch (bool, optional): Enable PyTorch acceleration (default: True).
            device (str or torch.device, optional): Computation device for PyTorch.
            method (str, optional): Decoding algorithm to use:
                - "euclidean": Brute-force exact search (default)
                - "torch_euclidean": GPU-accelerated exact search
                - "kdtree": K-d tree for fast exact search
                - "faiss": FAISS library for very fast approximate search
                - "interpolation": Spatial interpolation with anchor points
            **kwargs: Additional parameters passed to specific methods.

        Returns:
            Union[Trajectory, AgentState]: Decoded coordinates wrapped in appropriate
                dataclass. Shape matches input: (B,T,2) for trajectories, (B,2) for states.

        Examples:
            >>> # Decode place cell responses to trajectory
            >>> trajectory = sensory.decode_response(responses, method="kdtree")
            >>>
            >>> # Fast approximate decoding with FAISS
            >>> trajectory = sensory.decode_response(responses, method="faiss", n_clusters=50)
        """
        # Existing decoders operate on 2D maps; adapt tensor input explicitly.
        if self.arena.ndim != 2:
            raise ValueError('Response decoding currently requires a 2D arena.')
        if isinstance(response, torch.Tensor):
            response = response.detach()
            response = response.cpu()
            response = response.numpy()
        if isinstance(res_maps, torch.Tensor):
            res_maps = res_maps.detach()
            res_maps = res_maps.cpu()
            res_maps = res_maps.numpy()

        # Get response maps if not provided
        if res_maps is None:
            res_maps = self.aggregate_res_maps(keys, str_filter, type_filter)

        # Dispatch to appropriate implementation based on method
        if method == "kdtree":
            pred_coords, is_trajectory = decode_response_kdtree(response, res_maps)
        elif method == "torch_euclidean" and use_torch:
            chunk_size = kwargs.get('chunk_size', 1024)
            pred_coords, is_trajectory = decode_response_torch(response, res_maps, device, chunk_size)
        elif method == "faiss":
            n_clusters = kwargs.get('n_clusters', 100)
            pred_coords, is_trajectory = decode_response_faiss(response, res_maps, n_clusters)
        elif method == "interpolation":
            n_anchors = kwargs.get('n_anchors', 1000)
            random_state = kwargs.get('random_state', 42)
            pred_coords, is_trajectory = decode_response_interpolation(
                response, res_maps, n_anchors, random_state)
        else:
            # Default to euclidean method
            pred_coords, is_trajectory = decode_response_euclidean(response, res_maps)

        # Create and return appropriate dataclass
        coords = torch.as_tensor(pred_coords, device=self.gym.device)
        if is_trajectory:
            return Trajectory(coord=coords, device=self.gym.device)
        return AgentState(coord=coords, device=self.gym.device)

    def to(self, device, dtype=None):
        """Prepare spatial fields for repeated tensor queries."""
        for sensory in self.neuron_groups.values():
            if isinstance(sensory, SMBase):
                sensory.to(device, dtype=dtype)
        return self

    def get_response(self, agent_data, return_format='tensor', keys=None,
                     str_filter=None, type_filter=None, device=None):
        """Return selected responses as a dictionary, NumPy array, or tensor.

        Movement-modulated cells retain their NumPy equations; tensor requests
        convert those small outputs. Spatial cells query cached fields directly.
        """
        if return_format not in ('dict', 'array', 'tensor'):
            raise ValueError(f'Unknown return format: {return_format}')
        keys = self.filter_neurons(keys, str_filter, type_filter)
        responses = {}
        numpy_data = None
        tensor_input = isinstance(agent_data.coord, torch.Tensor)
        query_device = device
        if query_device is None and return_format in ('tensor', 'dict'):
            query_device = self.gym.device

        # Compute every selected response once, using each sensory's backend.
        for key in keys:
            sensory = self.neuron_groups[key]
            if isinstance(sensory, SMBase):
                output_format = 'tensor' if return_format == 'dict' else return_format
                response = sensory.get_response(agent_data, output_format, query_device)
            else:
                data = agent_data
                if tensor_input:
                    if numpy_data is None:
                        numpy_data = agent_data.as_numpy()
                    data = numpy_data
                response = sensory.get_response(data)
                if return_format in ('tensor', 'dict'):
                    response = torch.as_tensor(response, dtype=torch.float32, device=query_device)
            responses[key] = response
        if return_format == 'dict':
            return responses
        if not responses:
            raise ValueError('No sensory groups match the selection.')
        values = responses.values()
        values = list(values)
        if return_format == 'tensor':
            return torch.cat(values, dim=-1)
        return np.concatenate(values, axis=-1)

    def compute_res(self):
        for _sens in self.neuron_groups.values():
            _sens._compute_res()

    def save(self, file_path):
        """
        Save the sensory cells to a file.

        Args:
            file_path: Path to the file where the sensory cells will be saved.
        """
        all_sensory_data = {}
        for key, _sens in self.neuron_groups.items():
            all_sensory_data[key] = _sens.state_dict()
        with open(file_path, 'wb') as f:
            pickle.dump(all_sensory_data, f)

    def load_from_state_dict(self, state_dict, append=True):
        """
        Load the sensory cells from a state dictionary.

        Args:
            state_dict: State dictionary of the sensory cells.
            append: If True, append the sensory cells to the existing sensory cells.
                If False, replace the existing sensory cells.
        """
        if not append:
            self.neuron_groups = {}
        for key in state_dict.keys():
            _sens_data = state_dict[key]
            sensory_type = _sens_data.pop('sens_type')
            _sens_class = self._get_sensory_class(sensory_type)
            _sens = _sens_class.load_from_dict(_sens_data, self.arena)
            self.neuron_groups[key] = _sens

    def load(self, file_path):
        """
        Load the sensory cells from a file.

        Args:
            file_path: Path to the file where the sensory cells are saved.
        """
        with open(file_path, 'rb') as f:
            all_sensory_data = pickle.load(f)
        self.load_from_state_dict(all_sensory_data)
