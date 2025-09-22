import numpy as np
from .trajectory import Trajectory
from .raw_agent_state import RawAgentState
from typing import Optional


class RawTrajectory:
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
        self.coord[:, ts, :] = bhv_state.coord
        self.hd[:, ts, :] = bhv_state.hd
        self.mv_dir[:, ts, :] = bhv_state.mv_dir
        self.disp[:, ts, :] = bhv_state.disp

    def append(self, bhv_state: RawAgentState):
        self.coord.append(bhv_state.coord)
        self.disp.append(bhv_state.disp)

    def to_trajectory(self):
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
