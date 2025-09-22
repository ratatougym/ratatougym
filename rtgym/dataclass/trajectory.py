import os
import numpy as np
from typing import Optional
from .agent_state import AgentState


class Trajectory:
    """
    Class to hold trial data for the agent. This class is used to store the
    coordinates, coordinates in float, and head directions of the agent.
    """
    def __init__(
        self,
        coord: Optional[np.ndarray] = None,
        disp: Optional[np.ndarray] = None,
        hd: Optional[np.ndarray] = None
    ):
        """Initialize the Trajectory object with the given data.

        Args:
            coord: Coordinates of the agent as float values.
            disp: Displacements of the agent between coordinates.
            hd: Head directions of the agent.
        
        At least one of these must be provided.
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
            assert not_none_data[i].shape[0] == not_none_data[0].shape[0], "All provided data must have the same batch dimension"
            assert not_none_data[i].shape[1] == not_none_data[0].shape[1], "All provided data must have the same time dimension"

        # Store the input data
        self._coord = coord
        self.disp = disp
        self.hd = hd

    def copy(self):
        """
        Returns a copy of the Trajectory object.
        """
        return Trajectory(
            coord=self.coord.copy() if self.coord is not None else None,
            disp=self.disp.copy() if self.disp is not None else None,
            hd=self.hd.copy() if self.hd is not None else None
        )

    def __getitem__(self, index):
        """
        Returns the agent state at the specified time index.
        
        Args:
            time_index (int): The time index for which to get the agent state.
        
        Returns:
            AgentState: An AgentState object containing the agent state at the specified time index.
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
        Returns the number of time steps in the trajectory.
        """
        return self.coord.shape[1] if self.coord is not None else self.hd.shape[1] if self.hd is not None else self.disp.shape[1]

    def slice(self, start, end):
        return self.t_range((start, end))
    
    @property
    def float_coord(self):
        return self.coord
    
    @property
    def coord(self):
        return self._coord
    
    @coord.setter
    def coord(self, value):
        self._coord = value
    
    @property
    def int_coord(self):
        return self.coord.astype(int)

    @property
    def n_steps(self):
        """
        Returns the number of time steps in the trajectory.
        """
        return self.coord.shape[1] if self.coord is not None else self.hd.shape[1] if self.hd is not None else self.disp.shape[1]

    def reshape(self, shape):
        """
        Reshape the trajectory data to the specified shape.
        """
        assert isinstance(shape, tuple) and len(shape) == 2, "Shape must be a tuple of length 2"
        self.coord = self.coord.reshape(shape[0], shape[1], self.coord.shape[2])
        self.hd = self.hd.reshape(shape[0], shape[1], self.hd.shape[2])
        self.disp = self.disp.reshape(shape[0], shape[1], self.disp.shape[2])
        return self

    def t_range(self, range_):
        """
        Returns a new Trajectory object with data trimmed to the specified range.
        
        Args:
            range_ (list or tuple): A 2-element list or tuple specifying the start
                                    and end indices for trimming.
        
        Returns:
            Trajectory: A new Trajectory object with trimmed data.
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
        Load a Trajectory object from a file.

        Args:
            path (str): The file path to load the Trajectory object from.

        Returns:
            Trajectory: A Trajectory object loaded from the specified file.
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
        return cls(
            coord=state_dict['coord'],
            hd=state_dict['hd'],
            disp=state_dict['disp']
        )

    def state_dict(self):
        return {
            'coord': self.coord,
            'hd': self.hd,
            'disp': self.disp
        }
