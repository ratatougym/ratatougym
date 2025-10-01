import numpy as np
from .agent_state import AgentState


class RawAgentState:
    """Raw agent state data container for behavior generation and internal simulation.
    
    This class stores the complete internal state used during behavior generation,
    including both observable and hidden variables. Unlike AgentState which provides
    a cleaned interface for external use, RawAgentState maintains all intermediate
    variables needed for continuous agent simulation and behavior modeling.
    
    Args:
        coord (np.ndarray, optional): Agent coordinates in the arena space.
        hd (np.ndarray, optional): Head direction in radians.
        mv_dir (np.ndarray, optional): Movement direction vector.
        disp (np.ndarray, optional): Displacement vector from previous timestep.
        vel_norm (np.ndarray, optional): Velocity magnitude/norm.
        drift (np.ndarray, optional): Drift or bias component for movement.
        
    Note:
        This class is primarily used internally by behavior generation models
        and may contain intermediate computational states not exposed in the
        final AgentState output.
    """
    def __init__(self, coord=None, hd=None, mv_dir=None, disp=None, vel_norm=None, drift=None):
        self.coord = coord
        self.hd = hd
        self.mv_dir = mv_dir
        self.disp = disp
        self.vel_norm = vel_norm
        self.drift = drift

    @property
    def batch_size(self):
        """Get the batch size based on coordinate array dimensions.
        
        Returns the number of agents/samples in the current batch by examining
        the first dimension of the coordinate array. This is useful for
        vectorized operations across multiple agent instances.
        
        Returns:
            int: Number of agents in the current batch.
        """
        return self.coord.shape[0]
    
    @property
    def int_coord(self):
        """Get agent coordinates as integers for discrete grid indexing.
        
        Converts floating-point coordinates to integer values through truncation,
        which is useful for indexing into discrete spatial grids or arena maps
        during collision detection and spatial calculations.
        
        Returns:
            np.ndarray: Integer coordinates with shape (batch_size, 2).
        """
        return self.coord.astype(int)

    @property
    def float_coord(self):
        """Get agent coordinates as floating-point values.
        
        Returns the original floating-point coordinates without any conversion,
        preserving full precision for continuous spatial calculations and
        smooth movement tracking.
        
        Returns:
            np.ndarray: Floating-point coordinates with shape (batch_size, 2).
        """
        return self.coord

    def to_agent_state(self):
        """Convert raw internal state to standard AgentState format.
        
        Creates a cleaned AgentState instance containing only the essential
        observable variables (coordinates and displacement) while omitting
        internal computation variables. This provides a standardized interface
        for external systems that don't need access to raw behavioral state.
        
        Returns:
            AgentState: Cleaned agent state with coordinates and displacement data.
        """
        agent_state = AgentState(
            coord=self.coord,
            disp=self.disp
        )
        return agent_state

    def reset(self):
        """Reset all state variables to None for reinitialization.
        
        Clears all stored state variables by setting them to None. This is
        useful for resetting the agent state between episodes or when
        reinitializing the behavior generation system. After calling this
        method, new state values should be assigned before use.
        """
        self.coord = None
        self.disp = None
        self.vel_norm = None
        self.drift = None
        self.hd = None
        self.mv_dir = None

    def copy(self):
        """Create a deep copy of the current raw agent state.
        
        Returns a new RawAgentState instance with copied numpy arrays for
        all non-None state variables. This prevents unintended modifications
        to the original state when working with multiple state snapshots
        or during state rollbacks in behavior generation.
        
        Returns:
            RawAgentState: New instance with copied state data.
            
        Note:
            Only copies non-None arrays. None values remain as None in the copy.
        """
        return RawAgentState(
            coord=self.coord.copy(),
            disp=self.disp.copy(),
            vel_norm=self.vel_norm.copy(),
            drift=self.drift.copy(),
        )