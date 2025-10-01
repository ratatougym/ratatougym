"""
The neurons module defines the Neurons class, which manages groups of simulated 
neurons within an agent. These groups (e.g., place cells, grid cells, head 
direction cells) generate neuronal responses based on the agent's movement 
and environmental cues.

Neuron groups can be spatially modulated (e.g., place, boundary, grid cells) or 
movement modulated (e.g., speed, direction). Together they form the simulated 
neural substrate that links navigation with neuronal activity.
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


class Neurons:
    """
    Manages all simulated neuronal responses of the agent. 

    This class is automatically created by the `Agent` when a `RatatouGym` 
    environment is initialized. It should not be instantiated directly. 

    After initialization, the neuronal groups must be defined with a 
    **neuron_profiles** dictionary that specifies the simulated cell types 
    (e.g., place cells, grid cells, head direction cells) and their parameters.

    During spatial traversal, RatatouGym separates trajectory generation from 
    neuronal response computation. Once a trajectory is generated, the 
    `get_response` method computes the corresponding neuronal responses using 
    the defined tuning functions.

    Custom neuron groups can be added by placing their definitions in
    `rtgym/agent/neurons/spatial_modulated` or `movement_modulated`, adding 
    their class name to the `__all__` list of the appropriate `__init__.py`, 
    and registering them in `_get_neuron_class`.

    Args:
        gym (RatatouGym): Parent RatatouGym object.

    Examples:
        >>> from rtgym import RatatouGym
        >>> gym = RatatouGym(temporal_resolution=50, spatial_resolution=1)
        >>> # Set the neuronal groups
        >>> neuron_profiles = {
        >>>     'place_cells': {
        >>>         'type': 'place_cells',
        >>>         'n_cells': 100,
        >>>     },
        >>> }
        >>> gym.agent.set_neuron_profiles(neuron_profiles)
        >>> # Generate a trajectory
        >>> traj = agent.random_traverse(duration=10, batch_size=10)
        >>> # Get the neuronal responses
        >>> responses = agent.get_response(traj, return_format='array', keys=['place_cells'])
    """
    def __init__(self, gym):
        self.gym = gym
        self.neuron_groups = {}
        self.ranges = None  # Keep track of the indices of the simulated neuronal groups
    
    @property
    def common_params(self):
        return {'arena': self.gym.arena, 't_res': self.gym.t_res}

    def _on_arena_change(self):
        """
        Handle arena changes for all neuron groups.

        This method should update parameters of neuron groups that depend on
        arena geometry or resolution.
        """
        pass

    def list_all(self):
        """
        List all registered neuron groups.

        Returns:
            list: Names of all neuron groups currently defined.
        """
        return list(self.neuron_groups.keys())

    def init_from_profile(self, neuron_profiles):
        """
        Initialize neuron groups from a profile dictionary.

        Args:
            neuron_profiles (dict): Dictionary defining neuron groups and parameters.
        """
        self.neuron_groups = {}
        self._update_neurons(neuron_profiles if neuron_profiles is not None else {})
        self._update_ranges()
    
    def add_neuron_group(self, neuron_profiles: Dict[str, Any]):
        """
        Add a new neuron group to the existing configuration.

        Args:
            neuron_profile (dict): Dictionary defining a single neuron group.
        """
        self._update_neurons(neuron_profiles if neuron_profiles is not None else {})
        self._update_ranges()

    def _update_neurons(self, profile_list: Dict[str, Any]):
        """
        Internal helper to initialize neuron groups from a profile dictionary.
        """
        for key, value in profile_list.items():
            sensory_class = self._get_neuron_class(value['type'])
            self.neuron_groups[key] = sensory_class(
                sensory_key=key, 
                **self.common_params, 
                **value
            )

    def _update_ranges(self):
        _ranges = np.cumsum([_sens.n_cells for _sens in self.neuron_groups.values()])
        _ranges = np.insert(_ranges, 0, 0).tolist()
        self.ranges = {key: (_ranges[i], _ranges[i+1]) for i, key in enumerate(self.neuron_groups.keys())}
    
    def filter_neurons(self, keys=None, str_filter=None, type_filter=None):
        """
        Return neuron group keys that match given criteria.

        Priority of filters is: keys > str_filter > type_filter.

        Args:
            keys (str or list, optional): Explicit keys to select.
            str_filter (str, optional): Substring to match in neuron group names.
            type_filter (str, optional): Select by neuron type.

        Returns:
            list: Matching neuron group keys.
    """
        if keys is not None:
            if isinstance(keys, str):
                return_keys = [keys]
            elif isinstance(keys, list):
                return_keys = keys
            else:
                raise ValueError(f"Unknown keys: {keys}")
        elif str_filter is not None:
            return_keys = [key for key in self.neuron_groups.keys() if str_filter in key]
        elif type_filter is not None:
            return_keys = [key for key, neurons in self.neuron_groups.items() if type_filter == neurons.neuron_type]
        else:
            return_keys = list(self.neuron_groups.keys())
        return sorted(return_keys)

    def num_neurons(self, keys=None, str_filter=None, type_filter=None):
        """
        Count the total number of neurons across selected neuron groups.

        Returns:
            int: Number of neurons.
        """
        keys = self.filter_neurons(keys, str_filter, type_filter)
        return sum([self.neuron_groups[key].n_cells for key in keys])

    @staticmethod
    def _get_neuron_class(neuron_type):
        """
        Return the neuron class corresponding to the given type.

        This method maps a string type identifier to the appropriate class
        that implements the corresponding neuron group.

        Args:
            neuron_type (str): Neuron type identifier (must match one of the
                registered `cls.neuron_type` values).

        Returns:
            type: The neuron group class associated with the given type.

        Raises:
            AssertionError: If the type is not registered.
        """
        neuron_classes = {cls.neuron_type: cls for cls in 
                           [WeakSMCell, PlaceCell, BoundaryCell, GridCell,
                            SpeedCell, DirectionCell, DirectionRad, DisplacementAbs, 
                            HeadDirectionCell]}

        assert neuron_type in neuron_classes, f"Unknown neuron type: {neuron_type}"
        return neuron_classes[neuron_type]
    
    def aggregate_res_maps(
            self, keys: List[str] = None, str_filter: str = None, 
            type_filter: str = None):
        """
        Aggregate response maps from spatially modulated neuron groups.

        Combines response maps from multiple neuron groups into a single array.

        Args:
            keys (list, optional): Keys to filter the neuron_groups.
            str_filter (str, optional): String filter for neuron group names.
            type_filter (str, optional): Type filter for neuron groups.

        Returns:
            np.ndarray: Aggregated response maps with shape (n_cells, H, W).

        Raises:
            AssertionError: If non-spatially modulated neuron groups are included.
        """
        # Filter and aggregate the response maps.
        keys = self.filter_neurons(keys, str_filter, type_filter)
        res_maps = []
        for key in keys:
            assert self.neuron_groups[key].neuron_category == 'spatial_modulated', (
                "Only spatial_modulated cells can be decoded into a trajectory"
            )
            res_maps.append(self.neuron_groups[key].response_map)
        
        # Concatenate along the cell dimension. Shape: (n_cells, H, W)
        return np.concatenate(res_maps, axis=0)

    def decode_response(
            self, response: np.ndarray, res_maps: np.ndarray = None, 
            keys: List[str] = None, str_filter: str = None, type_filter: str = None,
            use_torch: bool = True, device: str = None, method: str = "euclidean", **kwargs):
        """
        Decode neuronal responses into spatial coordinates using various optimization methods.

        This method converts high-dimensional neuronal responses (e.g., from place cells,
        grid cells) back to spatial coordinates. Multiple algorithms are available,
        ranging from exact brute-force search to fast approximate methods.

        Args:
            response (np.ndarray): Neuronal response array of shape:
                - (B, T, D) for trajectory decoding
                - (B, D) for single state decoding
                where B=batch size, T=time steps, D=number of neurons.
            res_maps (np.ndarray, optional): Precomputed response maps.
                If None, computed from the selected neuron groups.
            keys (list, optional): Specific neuron group keys to include in decoding.
            str_filter (str, optional): String filter for neuron group names.
            type_filter (str, optional): Type filter for neuron groups.
            use_torch (bool, optional): Enable PyTorch acceleration (default: True).
            device (str or torch.device, optional): Computation device for PyTorch.
            method (str, optional):
                - "euclidean": Brute-force exact search (default)
                - "torch_euclidean": GPU-accelerated exact search
                - "kdtree": K-d tree for fast exact search
                - "faiss": FAISS library for approximate search
                - "interpolation": Spatial interpolation with anchor points
            **kwargs: Additional parameters for specific methods.

        Returns:
            Union[Trajectory, AgentState]: Decoded coordinates in the appropriate 
            dataclass, with shapes (B, T, 2) for trajectories and (B, 2) for states.

        Examples:
            >>> # Decode place cell responses to a trajectory
            >>> trajectory = neurons.decode_response(responses, method="kdtree")
            >>>
            >>> # Fast approximate decoding with FAISS
            >>> trajectory = neurons.decode_response(responses, method="faiss", n_clusters=50)
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
            return_format: str = 'dict', 
            keys: List[str] = None, 
            str_filter: str = None, 
            type_filter: str = None
        ):
        """
        Get neuronal responses for the given agent data.

        Args:
            agent_data: rtgym.dataclass.AgentState or rtgym.dataclass.Trajectory object.
            return_format: Format of the returned responses. Can be 'dict' or 'array'.
            keys: List of neuron group keys to get responses. If None, get responses for all.
            str_filter: Filter the neuron group keys by the given string.
            type_filter: Filter the neuron group keys by the given type.

        Returns:
            Union[dict, np.ndarray]: Neuronal responses. The responses are of shape 
            (n_cells, *arena_dimensions). After indexing, it will be of shape 
            (n_cells, n_batch). When  return_format is 'dict', it will be a 
            dictionary of responses. When return_format is 'array', it will be 
            a numpy array of responses.
        """
        # Set filter_keys if not provided
        keys = self.filter_neurons(keys, str_filter, type_filter)
        if return_format == 'dict':
            return {key: self.neuron_groups[key].get_response(agent_data) for key in keys}
        elif return_format == 'array':
            res_list = [self.neuron_groups[key].get_response(agent_data) for key in keys]
            return np.concatenate(res_list, axis=-1)
        else:
            raise ValueError(f"Unknown return format: {return_format}")

    def compute_res(self):
        """
        Some cell's response map might need to be computed after the arena has 
        been changed, this function serves to call all cell's compute_res method.
        """
        for _neuron in self.neuron_groups.values():
            _neuron._compute_res()

    def save(self, file_path):
        """
        Save the neuronal cells to a file.

        Args:
            file_path (str): Path to the file where the neuronal cells will be saved.
        """
        all_neuronal_data = {}
        for key, _neuron in self.neuron_groups.items():
            all_neuronal_data[key] = _neuron.state_dict()
        with open(file_path, 'wb') as f:
            pickle.dump(all_neuronal_data, f)

    def load_from_state_dict(self, state_dict, append=True):
        """
        Load the neuronal cells from a state dictionary.

        Args:
            state_dict (dict): State dictionary of the neurons.
            append: If True, append the neurons to the existing ones.
                If False, replace the existing neurons.
        """
        if not append:
            self.neuron_groups = {}
        for key in state_dict.keys():
            _neuron_data = state_dict[key]
            _neuron_class = Neurons._get_neuron_class(_neuron_data.pop('neuron_type'))
            _neuron = __neuron_class.load_from_dict(_neuron_data, self.gym.arena)
            self.neuron_groups[key] = _neuron

    def load(self, file_path):
        """
        Load the neuronal groups from a file.

        Args:
            file_path (str): Path to the file where the neuronal groups are saved.
        """
        with open(file_path, 'rb') as f:
            all_neuronal_data = pickle.load(f)
        self.load_from_state_dict(all_neuronal_data)
