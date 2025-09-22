import numpy as np


class AgentState:
    """Agent state data container for a single timestep.
    
    Holds the state data of the agent including coordinates, head directions,
    and displacements. This represents a snapshot of the agent's state at a 
    single timestep and can hold multiple batches of data.
    
    Args:
        coord (np.ndarray, optional): Integer coordinates of the agent.
        hd (np.ndarray, optional): Head direction of the agent in radians.
        disp (np.ndarray, optional): Displacement vector of the agent.
        
    Raises:
        ValueError: If all input parameters are None.
        
    Note:
        For unknown values, use np.nan as placeholder. This data class helps 
        standardize the data format for all sensory modalities.
    """
    
    def __init__(
        self,
        coord: np.ndarray = None,
        hd: np.ndarray = None,
        disp: np.ndarray = None
    ):
        # Ensure that at least one input is not None
        if coord is None and hd is None and disp is None:
            raise ValueError("At least one of coord, hd, or disp must be provided.")

        # Determine batch size based on the first non-None input
        batch_size = next(
            x.shape[0] for x in (coord, hd, disp) if x is not None
        )

        # Replace None inputs with np.nan arrays of appropriate shapes
        coord = coord if coord is not None else np.full((batch_size, 2), np.nan)
        hd = hd if hd is not None else np.full((batch_size, 1), np.nan)
        disp = disp if disp is not None else np.full((batch_size, 2), np.nan)

        # Check if the input data are all numpy arrays
        assert isinstance(coord, np.ndarray), "coord must be a numpy array"
        assert isinstance(hd, np.ndarray), "hd must be a numpy array"
        assert isinstance(disp, np.ndarray), "disp must be a numpy array"

        # Check if the input data all have two dimensions, (n_batch, n_features)
        assert coord.ndim == 2, "coord must have two dimensions, (n_batch, 2)"
        assert hd.ndim == 2, "hd must have two dimensions, (n_batch, 2)"
        assert disp.ndim == 2, "disp must have two dimensions, (n_batch, 2)"

        self.coord = coord
        self.hd = hd
        self.disp = disp

    @property
    def int_coord(self):
        return self.coord.astype(int)

    @property
    def float_coord(self):
        return self.coord
