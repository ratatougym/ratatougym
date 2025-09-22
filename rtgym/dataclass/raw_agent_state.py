import numpy as np
from .agent_state import AgentState


class RawAgentState:
    """
    This is the data class specifically designed to store the behavior generation state.
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
        return self.coord.shape[0]
    
    @property
    def int_coord(self):
        return self.coord.astype(int)

    @property
    def float_coord(self):
        return self.coord

    def to_agent_state(self):
        """
        Cast this internal state to a regular agent state.
        """
        agent_state = AgentState(
            coord=self.coord,
            disp=self.disp
        )
        return agent_state

    def reset(self):
        self.coord = None
        self.disp = None
        self.vel_norm = None
        self.drift = None
        self.hd = None
        self.mv_dir = None

    def copy(self):
        return RawAgentState(
            coord=self.coord.copy(),
            disp=self.disp.copy(),
            vel_norm=self.vel_norm.copy(),
            drift=self.drift.copy(),
        )