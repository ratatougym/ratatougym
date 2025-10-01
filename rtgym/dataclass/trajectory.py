import os
import numpy as np
from typing import Optional
from .agent_state import AgentState


class Trajectory:
    """
    Container for storing complete agent trajectory data over time.
    
    This class holds trial data for agents including coordinates, displacements,
    and head directions across multiple timesteps. It supports batch processing
    with multiple agents and provides convenient methods for data manipulation,
    slicing, and persistence. The trajectory data is stored as 3D numpy arrays
    with dimensions (n_batch, n_time, n_features).
    
    The class serves as the primary data structure for analyzing agent behavior
    patterns, storing experimental results, and interfacing with visualization
    and analysis tools throughout the rtgym ecosystem.
    """
    def __init__(
        self,
        coord: Optional[np.ndarray] = None,
        disp: Optional[np.ndarray] = None,
        hd: Optional[np.ndarray] = None
    ):
        """Initialize the Trajectory object with agent behavioral data.

        Creates a trajectory container from the provided agent state arrays.
        At least one data type must be provided, and all provided arrays must
        have consistent batch and time dimensions. The data is stored as 3D
        arrays to support vectorized operations across multiple agents and timesteps.

        Args:
            coord (np.ndarray, optional): Agent coordinates as float values with
                shape (n_batch, n_time, 2) representing [x, y] positions over time.
            disp (np.ndarray, optional): Displacement vectors between consecutive
                timesteps with shape (n_batch, n_time, 2) representing [dx, dy] movements.
            hd (np.ndarray, optional): Head direction angles in radians with
                shape (n_batch, n_time, 1) representing agent orientation over time.
        
        Raises:
            ValueError: If all input parameters are None.
            AssertionError: If inputs are not numpy arrays or have incompatible dimensions.
            
        Note:
            All provided arrays must have matching batch and time dimensions to ensure
            data consistency across different behavioral variables.
        """
        # Check if not all are None
        if coord is None and disp is None and dir is None:
            raise ValueError("At least one of coords_float, head_directions, or displacements must be provided")

        # Check if the input data are all numpy arrays
        assert isinstance(coord, np.ndarray) or coord is None, "coord must be a numpy array"
        assert isinstance(disp, np.ndarray) or disp is None, "disp must be a numpy array"
        assert isinstance(hd, np.ndarray) or hd is None, "hd must be a numpy array"

        # Check if the input data all have three dimensions, (n_batch, n_time, n_features)
        if coord is not None:
            assert coord.ndim == 3, "coord must have three dimensions (n_batch, n_time, n_features)"
        if disp is not None:
            assert disp.ndim == 3, "disp must have three dimensions (n_batch, n_time, n_features)"
        if hd is not None:
            assert hd.ndim == 3, "hd must have three dimensions (n_batch, n_time, n_features)"

        # Check if the first and second dimensions of the input data are the same for all not None
        data_list = [coord, disp, hd]
        not_none_data = [data for data in data_list if data is not None]
        
        for i in range(1, len(not_none_data)):
            assert not_none_data[i].shape[0] == not_none_data[0].shape[0], \
                "All provided data must have the same batch dimension"
            assert not_none_data[i].shape[1] == not_none_data[0].shape[1], \
                "All provided data must have the same time dimension"

        # Store the input data
        self._coord = coord
        self.disp = disp
        self.hd = hd

    def copy(self):
        """
        Create a deep copy of the Trajectory object.
        
        Returns a new Trajectory instance with independently copied data arrays.
        This ensures that modifications to the copy do not affect the original
        trajectory data, enabling safe data manipulation and analysis workflows.
        
        Returns:
            Trajectory: A new Trajectory object with copied data arrays containing
                the same behavioral data as the original instance.
        """
        return Trajectory(
            coord=self.coord.copy() if self.coord is not None else None,
            disp=self.disp.copy() if self.disp is not None else None,
            hd=self.hd.copy() if self.hd is not None else None
        )

    def __getitem__(self, index):
        """
        Extract agent state or trajectory slice using indexing syntax.
        
        Supports both integer indexing to get AgentState at a specific timestep
        and slice indexing to get a sub-trajectory. This provides convenient
        access to temporal subsets of the trajectory data for analysis and
        visualization purposes.
        
        Args:
            index (int or slice): Time index for extracting data. Integer indices
                return AgentState objects for a single timestep, while slice objects
                return new Trajectory objects covering the specified time range.
        
        Returns:
            AgentState or Trajectory: AgentState object for integer indices containing
                agent data at the specified timestep, or Trajectory object for slice
                indices containing data over the specified time range.
                
        Raises:
            TypeError: If index is neither int nor slice.
        """
        if isinstance(index, int):
            return AgentState(
                coord=self.coord[:, index, :].copy() if self.coord is not None else None,
                hd=self.hd[:, index, :].copy() if self.hd is not None else None,
                disp=self.disp[:, index, :].copy() if self.disp is not None else None
            )
        elif isinstance(index, slice):
            return Trajectory(
                coord=self.coord[:, index, :].copy() if self.coord is not None else None,
                hd=self.hd[:, index, :].copy() if self.hd is not None else None,
                disp=self.disp[:, index, :].copy() if self.disp is not None else None
            )
        else:
            raise TypeError("Index must be an int or slice")

    def __len__(self):
        """
        Get the number of timesteps in the trajectory.
        
        Returns the temporal length of the trajectory by examining the time
        dimension of the available data arrays. This provides a convenient
        way to determine trajectory duration for iteration and analysis.
        
        Returns:
            int: Number of timesteps in the trajectory data.
        """
        return self.coord.shape[1] if self.coord is not None else self.hd.shape[1] if self.hd is not None else self.disp.shape[1]

    def slice(self, start, end):
        """
        Create a trajectory slice between specified timestep indices.
        
        Convenience method that calls t_range with the provided start and end
        indices. This provides a simple interface for extracting temporal
        subsets of trajectory data for focused analysis or visualization.
        
        Args:
            start (int): Starting timestep index (inclusive).
            end (int): Ending timestep index (exclusive).
            
        Returns:
            Trajectory: New trajectory object containing data from start to end timesteps.
        """
        return self.t_range((start, end))
    
    @property
    def float_coord(self):
        """
        Get agent coordinates as floating-point values.
        
        Returns the stored coordinate data in its original floating-point format,
        preserving full precision for accurate spatial calculations and continuous
        movement analysis. This is an alias for the coord property.
        
        Returns:
            np.ndarray: Floating-point coordinates with shape (n_batch, n_time, 2).
        """
        return self.coord
    
    @property
    def coord(self):
        """
        Get the stored coordinate data.
        
        Provides access to the agent coordinate arrays stored in the trajectory.
        Coordinates represent agent positions over time and are fundamental
        for spatial analysis and visualization of agent behavior.
        
        Returns:
            np.ndarray or None: Coordinate array with shape (n_batch, n_time, 2)
                if coordinates were provided during initialization, None otherwise.
        """
        return self._coord
    
    @coord.setter
    def coord(self, value):
        """
        Set new coordinate data for the trajectory.
        
        Updates the internal coordinate storage with new positional data.
        This setter allows for dynamic modification of trajectory coordinates
        while maintaining the internal data structure consistency.
        
        Args:
            value (np.ndarray or None): New coordinate array with appropriate
                shape or None to clear coordinate data.
        """
        self._coord = value
    
    @property
    def int_coord(self):
        """
        Get agent coordinates converted to integer values.
        
        Converts floating-point coordinates to integers through truncation,
        which is useful for indexing into discrete spatial grids, arena maps,
        or other discrete spatial representations during analysis.
        
        Returns:
            np.ndarray: Integer coordinates with shape (n_batch, n_time, 2)
                containing truncated coordinate values.
        """
        return self.coord.astype(int)

    @property
    def n_steps(self):
        """
        Get the number of timesteps in the trajectory.
        
        Returns the temporal duration of the trajectory by examining the time
        dimension of available data arrays. This property provides consistent
        access to trajectory length regardless of which data types are present.
        
        Returns:
            int: Number of timesteps in the trajectory data.
        """
        return self.coord.shape[1] if self.coord is not None else self.hd.shape[1] if self.hd is not None else self.disp.shape[1]

    def reshape(self, shape):
        """
        Reshape trajectory data arrays to new batch and time dimensions.
        
        Modifies the trajectory data in-place by reshaping all available arrays
        to the specified batch and time dimensions. This is useful for changing
        the data organization for different analysis or processing requirements
        while preserving the feature dimensions.
        
        Args:
            shape (tuple): Two-element tuple specifying new (n_batch, n_time) dimensions.
                The feature dimension is preserved automatically for each data type.
                
        Returns:
            Trajectory: Self-reference to enable method chaining.
            
        Raises:
            AssertionError: If shape is not a 2-element tuple.
            
        Note:
            This method modifies the trajectory data in-place. Use copy() first
            if you need to preserve the original data structure.
        """
        assert isinstance(shape, tuple) and len(shape) == 2, "Shape must be a tuple of length 2"
        self.coord = self.coord.reshape(shape[0], shape[1], self.coord.shape[2])
        self.hd = self.hd.reshape(shape[0], shape[1], self.hd.shape[2])
        self.disp = self.disp.reshape(shape[0], shape[1], self.disp.shape[2])
        return self

    def t_range(self, range_):
        """
        Extract a temporal subset of the trajectory data.
        
        Creates a new Trajectory object containing data from the specified
        time range. This is useful for analyzing specific portions of agent
        behavior, removing initialization periods, or focusing on particular
        experimental phases.
        
        Args:
            range_ (list or tuple): Two-element sequence specifying [start, end)
                timestep indices. Start is inclusive, end is exclusive following
                Python slicing conventions.
        
        Returns:
            Trajectory: New Trajectory object containing data from the specified
                time range with all available data types trimmed consistently.
                
        Raises:
            AssertionError: If range parameters are invalid (wrong length, invalid
                ordering, or out of bounds).
        """
        # Check if the range is valid
        assert len(range_) == 2, "range_ must be a tuple of length 2"
        assert range_[0] < range_[1], "range_[0] must be less than range_[1]"
        assert range_[1] <= self.coord.shape[1], "range_[1] must be less than the trial duration"
        assert range_[0] >= 0, "range_[0] must be greater than or equal to 0"

        # Trim the data
        return Trajectory(
            coord=self.coord[:, range_[0]:range_[1]] if self.coord is not None else None,
            hd=self.hd[:, range_[0]:range_[1]] if self.hd is not None else None,
            disp=self.disp[:, range_[0]:range_[1]] if self.disp is not None else None
        )

    @staticmethod
    def load(path):
        """
        Load a Trajectory object from a compressed numpy archive file.
        
        Restores trajectory data from an NPZ file created by numpy.savez.
        This enables persistence of trajectory data across sessions and
        sharing of experimental results between different analysis workflows.
        The method automatically handles missing data types gracefully.

        Args:
            path (str): File path to the NPZ archive containing trajectory data.
                Must have '.npz' extension and contain trajectory arrays with
                standard naming conventions.

        Returns:
            Trajectory: A new Trajectory object loaded with data from the file.
            
        Raises:
            AssertionError: If the file does not have '.npz' extension.
            FileNotFoundError: If the specified file does not exist.
        """
        assert os.path.splitext(path)[1] == '.npz', "File extension must be .npz"
        loaded_dict = np.load(path)
        return Trajectory(
            coord=loaded_dict['coord'] if 'coord' in loaded_dict else None,
            hd=loaded_dict['hd'] if 'hd' in loaded_dict else None,
            disp=loaded_dict['disp'] if 'disp' in loaded_dict else None
        )

    @classmethod
    def from_dict(cls, state_dict):
        """
        Create a Trajectory object from a dictionary of arrays.
        
        Constructs a new Trajectory instance using data stored in a dictionary
        format. This is useful for interfacing with serialization systems,
        configuration files, or other data sources that organize trajectory
        data as key-value pairs.
        
        Args:
            state_dict (dict): Dictionary containing trajectory data arrays with
                keys 'coord', 'hd', and 'disp' corresponding to coordinates,
                head directions, and displacements respectively.
                
        Returns:
            Trajectory: New Trajectory object initialized with the dictionary data.
        """
        return cls(
            coord=state_dict['coord'],
            hd=state_dict['hd'],
            disp=state_dict['disp']
        )

    def state_dict(self):
        """
        Export trajectory data as a dictionary.
        
        Creates a dictionary representation of the trajectory data suitable
        for serialization, debugging, or interfacing with external systems.
        The returned dictionary contains all available trajectory data arrays
        with standard key names.
        
        Returns:
            dict: Dictionary containing trajectory data with keys 'coord', 'hd',
                and 'disp' mapping to their respective numpy arrays or None values.
        """
        return {
            'coord': self.coord,
            'hd': self.hd,
            'disp': self.disp
        }
