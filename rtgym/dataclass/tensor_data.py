"""Tensor data for EMA navigation and device-resident sensory queries."""

import torch
from .agent_state import AgentState
from .trajectory import Trajectory


class TensorAgentState(AgentState):
    """EMA state. Coordinates and speeds use grid cells and grid cells/step.

    ``head_dir`` is a unit vector; ``hd`` exposes the legacy 2D angle convention.
    Target fields retain the EMA generator state when continuing a trajectory.
    """

    fields = ('coord', 'spd', 'spd_target', 'mv_dir', 'mv_dir_target', 'head_dir')

    def __init__(self, coord=None, spd=None, mv_dir=None, head_dir=None,
                 spd_target=None, mv_dir_target=None, device='cpu'):
        self.device = torch.device(device)
        values = (coord, spd, spd_target, mv_dir, mv_dir_target, head_dir)
        for name, value in zip(self.fields, values):
            if value is not None:
                value = torch.as_tensor(value, device=self.device)
            setattr(self, name, value)

    @property
    def int_coord(self):
        return self.coord.long()

    @property
    def disp(self):
        if self.spd is None or self.mv_dir is None:
            return None
        return self.spd * self.mv_dir

    @property
    def hd(self):
        if self.head_dir is None:
            return None
        if self.head_dir.shape[-1] != 2:
            raise ValueError('hd angles require 2D data; use head_dir in 3D.')
        angle = torch.atan2(self.head_dir[..., 0], self.head_dir[..., 1])
        return angle.unsqueeze(-1)

    def clone(self):
        values = {}
        for name in self.fields:
            value = getattr(self, name)
            if value is not None:
                values[name] = value.clone()
        return TensorAgentState(**values, device=self.device)

    copy = clone

    def to(self, device):
        """Return a state on the requested device without moving this state."""
        values = {}
        for name in self.fields:
            value = getattr(self, name)
            if value is not None:
                values[name] = value.to(device)
        return TensorAgentState(**values, device=device)

    def to_numpy(self):
        if self.coord.shape[-1] != 2:
            raise ValueError('Legacy AgentState conversion requires 2D data.')
        coord = self.coord.detach()
        coord = coord.cpu()
        disp = self.disp.detach()
        disp = disp.cpu()
        hd = self.hd.detach()
        hd = hd.cpu()
        coord = coord.numpy()
        disp = disp.numpy()
        hd = hd.numpy()
        return AgentState(coord=coord, disp=disp, hd=hd)


class TensorTrajectory(Trajectory):
    """Batched tensor trajectory with the official time-first indexing API.

    ``traj[t]`` selects a state; ``traj[:, start:stop]`` selects batches and time.
    ``disp`` is commanded motion, which may differ from actual motion at walls.
    """

    fields = ('coord', 'spd', 'mv_dir', 'head_dir')

    def __init__(self, coord, spd, mv_dir, head_dir, device='cpu'):
        self.device = torch.device(device)
        values = (coord, spd, mv_dir, head_dir)
        for name, value in zip(self.fields, values):
            value = torch.as_tensor(value, device=self.device)
            if value.ndim != 3 or value.shape[:2] != coord.shape[:2]:
                raise ValueError('Trajectory fields must share batch and time dimensions.')
            setattr(self, name, value)

    @property
    def int_coord(self):
        return self.coord.long()

    @property
    def size(self):
        return self.coord.shape[:2]

    @property
    def disp(self):
        return self.spd * self.mv_dir

    @property
    def hd(self):
        if self.head_dir.shape[-1] != 2:
            raise ValueError('hd angles require 2D data; use head_dir in 3D.')
        angle = torch.atan2(self.head_dir[..., 0], self.head_dir[..., 1])
        return angle.unsqueeze(-1)

    def copy(self):
        values = {}
        for name in self.fields:
            value = getattr(self, name)
            values[name] = value.clone()
        return TensorTrajectory(**values, device=self.device)

    def __getitem__(self, index):
        batch_index = slice(None)
        time_index = index
        if isinstance(index, tuple):
            batch_index, time_index = index
        if isinstance(batch_index, int):
            batch_index = [batch_index]

        # Select axes separately so batch and time lists form a Cartesian product.
        values = {}
        for name in self.fields:
            value = getattr(self, name)
            selected = value[batch_index]
            selected = selected[:, time_index]
            values[name] = selected.clone()
        if isinstance(time_index, int):
            return TensorAgentState(**values, device=self.device)
        return TensorTrajectory(**values, device=self.device)

    def to(self, device):
        """Return a trajectory on the requested device."""
        values = {}
        for name in self.fields:
            value = getattr(self, name)
            values[name] = value.to(device)
        return TensorTrajectory(**values, device=device)

    def to_numpy(self):
        if self.coord.shape[-1] != 2:
            raise ValueError('Legacy Trajectory conversion requires 2D data.')
        coord = self.coord.detach()
        coord = coord.cpu()
        disp = self.disp.detach()
        disp = disp.cpu()
        hd = self.hd.detach()
        hd = hd.cpu()
        coord = coord.numpy()
        disp = disp.numpy()
        hd = hd.numpy()
        return Trajectory(coord=coord, disp=disp, hd=hd)

    def reshape(self, shape):
        if len(shape) != 2:
            raise ValueError('Shape must contain batch and time dimensions.')
        for name in self.fields:
            value = getattr(self, name)
            value = value.reshape(*shape, value.shape[-1])
            setattr(self, name, value)
        return self

    def t_range(self, range_):
        start, stop = range_
        if not 0 <= start < stop <= self.n_steps:
            raise ValueError('Invalid time range.')
        return self[:, start:stop]

    def state_dict(self):
        return {name: getattr(self, name) for name in self.fields}

    @classmethod
    def from_dict(cls, state_dict):
        coord = state_dict['coord']
        return cls(**state_dict, device=coord.device)

    @staticmethod
    def load(path):
        raise ValueError('Use TensorTrajectory.from_dict with tensor fields.')
