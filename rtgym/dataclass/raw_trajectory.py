import numpy as np
from .trajectory import Trajectory
from .raw_agent_state import RawAgentState
from typing import Optional


class RawTrajectory:
    """
    Raw trajectory is a data container class. This only serve as an intermedicate
    data containing class during the generation of the agent behavior (trajectory).

    This class stores sequences of raw agent states during behavior generation,
    supporting both autonomous (pre-allocated) and controllable (dynamic) modes.
    It maintains internal behavioral variables like movement direction that are
    not exposed in the final Trajectory output but are needed during simulation.

    Args:
        dur_ts (int, optional): Duration in timesteps for autonomous behavior.
            When provided with batch_size, creates pre-allocated arrays.
        batch_size (int, optional): Number of agents in the batch for autonomous
            behavior. When provided with dur_ts, creates pre-allocated arrays.

    Note:
        This class should not be used directly by the user.
        If both dur_ts and batch_size are provided, the trajectory operates in
        "autonomous" mode with pre-allocated numpy arrays. If neither is provided,
        it operates in "controllable" mode using dynamic lists for data collection.
    """
    
    def __init__(self, dur_ts: Optional[int] = None, batch_size: Optional[int] = None):
        if dur_ts is not None and batch_size is not None:
            self.behavior_type = "autonomous"
            self.dur_ts = dur_ts
            self.batch_size = batch_size
            self.coord = np.zeros((batch_size, dur_ts, 2))
            self.hd = np.zeros((batch_size, dur_ts, 1))
            self.mv_dir = np.zeros((batch_size, dur_ts, 1))
            self.disp = np.zeros((batch_size, dur_ts, 2))
        else:
            self.behavior_type = "controllable"
            self.coord = []
            self.hd = []
            self.mv_dir = []
            self.disp = []

    def update_state(self, ts, bhv_state: RawAgentState):
        """
        Update trajectory state at a specific timestep for autonomous agent behavior.

        Updates all agent states at the given timestep with data from the provided
        RawAgentState. This method is used in autonomous mode where trajectory
        data is stored in pre-allocated arrays indexed by timestep.

        Args:
            ts (int): Timestep index to update in the trajectory arrays.
            bhv_state (RawAgentState): Raw agent state containing coordinates,
                head direction, movement direction, and displacement data.
        """
        self.coord[:, ts, :] = bhv_state.coord
        self.hd[:, ts, :] = bhv_state.hd
        self.mv_dir[:, ts, :] = bhv_state.mv_dir
        self.disp[:, ts, :] = bhv_state.disp

    def append(self, bhv_state: RawAgentState):
        """
        Append agent state to trajectory for controllable agent behavior.

        Dynamically adds a new timestep of agent data to the trajectory lists.
        This method is used in controllable mode where trajectory length is
        not predetermined and data is collected incrementally.

        Args:
            bhv_state (RawAgentState): Raw agent state containing coordinates
                and displacement data to append to the trajectory.
  
        Note:
            Only coordinates and displacement are collected in controllable mode,
            as head direction and movement direction are typically not needed
            for externally controlled behavior.
        """
        self.coord.append(bhv_state.coord)
        self.disp.append(bhv_state.disp)

    def to_trajectory(self):
        """
        Convert raw trajectory to standard Trajectory format.

        Transforms the raw trajectory data into a cleaned Trajectory instance
        suitable for external use. The conversion process depends on the behavior
        type and handles appropriate data reshaping and filtering.

        Returns:
            Trajectory: Cleaned trajectory containing only essential observable
                variables (coordinates, head direction, displacement) with
                proper array dimensions for downstream processing.

        Note:
            For autonomous mode, data is used directly from pre-allocated arrays.
            For controllable mode, lists are converted to numpy arrays and
            transposed to match the expected (batch_size, timesteps, features) format.
        """
        if self.behavior_type == "autonomous":  
            return Trajectory(
                coord=self.coord,
                hd=self.hd,
                disp=self.disp
            )
        else:
            return Trajectory(
                coord=np.array(self.coord).transpose(1, 0, 2),
                disp=np.array(self.disp).transpose(1, 0, 2)
            )
