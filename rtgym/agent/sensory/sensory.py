"""
The sensory module contains the Sensory class which is responsible for 
creating and managing spatially and movement-modulated sensory cells of the agent.
"""


import rtgym
import pickle
import numpy as np
import torch
from typing import Dict, Any, Union, List, Tuple

from rtgym.dataclass import AgentState, Trajectory
from rtgym.utils.decode_response import (
    decode_response_euclidean,
    decode_response_kdtree,
    decode_response_torch,
    decode_response_faiss,
    decode_response_interpolation,
    create_dataclass_result
)
from .spatial_modulated import *
from .movement_modulated import *


class Sensory:
    """
    The class object that manages the sensory system of the agent. More broadly, 
    it handles all simulated neuronal responses of the agent.  

    When the gym is initialized, an `Agent` object is automatically created, 
    which in turn creates a `Sensory` object. The `Sensory` object is initially 
    just a placeholder and must be initialized with a sensory profile (a dictionary) 
    that defines the simulated neuronal groups and their parameters.  

    During spatial traversal, RatatouGym separates the concerns of trajectory 
    generation and neuronal response computation. Once a trajectory is generated, 
    RatatouGym calls the `get_response` method of the `Sensory` object. This 
    method takes the trajectory as input and computes the corresponding neuronal 
    responses using the defined tuning curves.  

    This class should not be initialized directly. The `RatatouGym` class will 
    automatically manage it.

    Args:
        gym (RatatouGym): Parent RatatouGym object.
    """
    def __init__(self, gym):
        self.gym = gym
        self.sensories = {}
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
        return list(self.sensories.keys())

    def init_from_profile(self, sensory_profile):
        # Initialize spatial and movement modulated cells
        self.sensories = {}
        self._update_sensories(sensory_profile if sensory_profile is not None else {})
        self._update_ranges()
    
    def add_sensory(self, sensory_profile: Dict[str, Any]):
        """
        Add a sensory cell to the sensory system.

        Args:
            sensory_profile: Dictionary containing the sensory profile.
        """
        self._update_sensories(sensory_profile if sensory_profile is not None else {})
        self._update_ranges()

    def _update_sensories(self, profile_list):
        """
        Initializes the cells based on the provided profile list.
        """
        for key, value in profile_list.items():
            sensory_class = Sensory._get_sensory_class(value['type'])
            self.sensories[key] = sensory_class(
                sensory_key=key, 
                **self.common_params, 
                **value
            )
    
    def _update_ranges(self):
        _ranges = np.cumsum([_sens.n_cells for _sens in self.sensories.values()])
        _ranges = np.insert(_ranges, 0, 0).tolist()
        self.ranges = {key: (_ranges[i], _ranges[i+1]) for i, key in enumerate(self.sensories.keys())}
    
    def filter_sensories(self, keys=None, str_filter=None, type_filter=None):
        """
        This helps to find the keys of the sensory cells that match the given criteria.

        It will prioritize the most specific filter. The specificity from most to least is:
            keys > str_filter > type_filter
        """
        if keys is not None:
            if isinstance(keys, str):
                return_keys = [keys]
            elif isinstance(keys, list):
                return_keys = keys
            else:
                raise ValueError(f"Unknown keys: {keys}")
        elif str_filter is not None:
            return_keys = [key for key in self.sensories.keys() if str_filter in key]
        elif type_filter is not None:
            return_keys = [key for key, sensory_item in self.sensories.items() if type_filter == sensory_item.sens_type]
        else:
            return_keys = list(self.sensories.keys())
        return sorted(return_keys)

    def num_sensories(self, keys=None, str_filter=None, type_filter=None):
        keys = self.filter_sensories(keys, str_filter, type_filter)
        return sum([self.sensories[key].n_cells for key in keys])

    @staticmethod
    def _get_sensory_class(sensory_type):
        sensory_classes = {cls.sens_type: cls for cls in 
                           [WeakSMCell, PlaceCell, BoundaryCell, GridCell,
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
        keys = self.filter_sensories(keys, str_filter, type_filter)
        res_maps = []
        for key in keys:
            assert self.sensories[key].sens_category == 'spatial_modulated', (
                "Only spatial_modulated sensory cells can be decoded into a trajectory"
            )
            res_maps.append(self.sensories[key].response_map)
        
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
            response (np.ndarray): Sensory response array of shape:
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
        # Get response maps if not provided
        if res_maps is None:
            res_maps = self.aggregate_res_maps(keys, str_filter, type_filter)

        # Dispatch to appropriate implementation based on method
        if method == "kdtree":
            pred_coords, is_trajectory = decode_response_kdtree(response, res_maps)
        elif method == "torch_euclidean" and use_torch:
            pred_coords, is_trajectory = decode_response_torch(
                response, res_maps, device, kwargs.get('chunk_size', 1024)
            )
        elif method == "faiss":
            pred_coords, is_trajectory = decode_response_faiss(
                response, res_maps, kwargs.get('n_clusters', 100)
            )
        elif method == "interpolation":
            pred_coords, is_trajectory = decode_response_interpolation(
                response, res_maps,
                kwargs.get('n_anchors', 1000),
                kwargs.get('random_state', 42)
            )
        else:
            # Default to euclidean method
            pred_coords, is_trajectory = decode_response_euclidean(response, res_maps)

        # Create and return appropriate dataclass
        return create_dataclass_result(pred_coords, is_trajectory)

    def get_response(
            self, 
            agent_data: Union[AgentState, Trajectory],
            return_format='dict', 
            keys=None, 
            str_filter=None, 
            type_filter=None
        ):
        """
        Get sensory responses for the given trajectory.

        Args:
            traj: rtgym.dataclass.Trajectory object.
            return_format: Format of the returned responses. Can be 'dict' or 'array'.
            keys: List of sensory keys to get responses. If None, get responses for all.
            str_filter: Filter the sensory keys by the given string.
            type_filter: Filter the sensory keys by the given type.

        Returns:
            responses: Sensory responses. The responses are of shape (n_cells, *arena_dimensions).
                After indexing, it will be of shape (n_cells, n_batch).
                When return_format is 'dict', it will be a dictionary of responses.
                When return_format is 'array', it will be a numpy array of responses.
        """
        # Set filter_keys if not provided
        keys = self.filter_sensories(keys, str_filter, type_filter)
        if return_format == 'dict':
            return {key: self.sensories[key].get_response(agent_data) for key in keys}
        elif return_format == 'array':
            res_list = [self.sensories[key].get_response(agent_data) for key in keys]
            return np.concatenate(res_list, axis=-1)
        else:
            raise ValueError(f"Unknown return format: {return_format}")

    def compute_res(self):
        for _sens in self.sensories.values():
            _sens._compute_res()

    def save(self, file_path):
        """
        Save the sensory cells to a file.

        Args:
            file_path: Path to the file where the sensory cells will be saved.
        """
        all_sensory_data = {}
        for key, _sens in self.sensories.items():
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
            self.sensories = {}
        for key in state_dict.keys():
            _sens_data = state_dict[key]
            _sens_class = self._get_sensory_class(_sens_data.pop('sens_type'))
            _sens = _sens_class.load_from_dict(_sens_data, self.arena)
            self.sensories[key] = _sens

    def load(self, file_path):
        """
        Load the sensory cells from a file.

        Args:
            file_path: Path to the file where the sensory cells are saved.
        """
        with open(file_path, 'rb') as f:
            all_sensory_data = pickle.load(f)
        self.load_from_state_dict(all_sensory_data)
