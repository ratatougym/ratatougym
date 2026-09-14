"""Tensor data for EMA navigation and device-resident sensory queries."""

import torch
from .dataclass.agent_state import AgentState as NumPyAgentState
from .dataclass.trajectory import Trajectory as NumPyTrajectory


class AgentState(NumPyAgentState):
    """EMA state. Coordinates and speeds use grid cells and grid cells/step.

    ``head_dir`` is a unit vector; ``hd`` exposes the legacy 2D angle convention.
    Target fields retain the EMA generator state when continuing a trajectory.
    """

    fields = ('coord', 'spd', 'spd_target', 'mv_dir', 'mv_dir_target', 'head_dir')

    def __init__(self, coord=None, spd=None, spd_target=None, mv_dir=None,
                 mv_dir_target=None, head_dir=None, device='cpu'):
        self.device = torch.device(device)
        values = (coord, spd, spd_target, mv_dir, mv_dir_target, head_dir)
        for name, value in zip(self.fields, values):
            if value is not None:
                value = torch.as_tensor(value, device=self.device)
            setattr(self, name, value)

    @property
    def int_coord(self):
        if self.coord is None:
            return None
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
        return AgentState(**values, device=self.device)

    copy = clone

    @property
    def coord(self):
        return self._coord

    @coord.setter
    def coord(self, value):
        if value is not None:
            value = torch.as_tensor(value, device=self.device)
        self._coord = value

    def reset(self):
        for name in self.fields:
            setattr(self, name, None)

    def to(self, device):
        """Move this state in place, matching grid_and_place."""
        self.device = torch.device(device)
        for name in self.fields:
            value = getattr(self, name)
            if value is not None:
                value = value.to(self.device)
                setattr(self, name, value)
        return self

    def to_numpy(self):
        """Return coordinate, head-vector and displacement arrays."""
        arrays = []
        for value in (self.coord, self.head_dir, self.disp):
            if value is not None:
                value = value.detach()
                value = value.cpu()
                value = value.numpy()
            arrays.append(value)
        return tuple(arrays)

    def as_numpy(self):
        """Adapt 2D tensor data for the retained NumPy utilities."""
        if self.coord is not None and self.coord.shape[-1] != 2:
            raise ValueError('NumPy data conversion requires 2D coordinates.')
        values = {'coord': self.coord, 'disp': self.disp, 'hd': self.hd}
        for name, value in values.items():
            if value is not None:
                value = value.detach()
                value = value.cpu()
                values[name] = value.numpy()
        return NumPyAgentState(**values)



class Trajectory(NumPyTrajectory):
    """Batched tensor trajectory following grid_and_place indexing.

    ``traj[b]`` selects a batch; ``traj[:, t]`` selects time across batches.
    ``disp`` is commanded motion, which may differ from actual motion at walls.
    """

    fields = ('coord', 'spd', 'mv_dir', 'head_dir')

    def __init__(self, coord=None, head_dir=None, spd=None, mv_dir=None, device='cpu'):
        self.device = torch.device(device)
        values = (coord, spd, mv_dir, head_dir)
        shape = None
        for name, value in zip(self.fields, values):
            if value is not None:
                value = torch.as_tensor(value, device=self.device)
                if value.ndim != 3:
                    raise ValueError('Trajectory fields must have batch, time and feature axes.')
                if shape is not None and value.shape[:2] != shape:
                    raise ValueError('Trajectory fields must share batch and time dimensions.')
                shape = value.shape[:2]
            setattr(self, name, value)

    def __len__(self):
        if self.coord is None:
            raise ValueError('Coordinate data is not set.')
        return self.coord.shape[1]

    @property
    def n_steps(self):
        return len(self)

    @property
    def int_coord(self):
        if self.coord is None:
            return None
        return self.coord.long()

    @property
    def size(self):
        return self.coord.shape[:2]

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

    def copy(self):
        values = {}
        for name in self.fields:
            value = getattr(self, name)
            if value is not None:
                values[name] = value.clone()
        return Trajectory(**values, device=self.device)

    def __getitem__(self, index):
        # A single index selects batches; a tuple selects batches and time.
        select_time = isinstance(index, tuple)
        batch_index, time_index = index if select_time else (index, slice(None))
        if isinstance(batch_index, int):
            batch_index = [batch_index]
        if isinstance(time_index, int):
            time_index = [time_index]
        if batch_index is Ellipsis:
            batch_index = slice(None)
        if time_index is Ellipsis:
            time_index = slice(None)

        # Separate indexing keeps the Cartesian product of batch and time lists.
        values = {}
        selected_shape = None
        for name in self.fields:
            value = getattr(self, name)
            if value is not None:
                selected = value[batch_index]
                selected = selected[:, time_index]
                values[name] = selected.clone()
                selected_shape = selected.shape
        if selected_shape is None:
            raise ValueError('Cannot index an empty trajectory.')
        if select_time and selected_shape[1] == 1 and selected_shape[0] >= 1:
            for name, value in values.items():
                values[name] = value.squeeze(1)
            return AgentState(**values, device=self.device)
        return Trajectory(**values, device=self.device)

    def to(self, device):
        """Return a trajectory on the requested device."""
        values = {}
        for name in self.fields:
            value = getattr(self, name)
            if value is not None:
                values[name] = value.to(device)
        return Trajectory(**values, device=device)

    def as_numpy(self):
        """Adapt 2D tensor data for the retained NumPy utilities."""
        if self.coord is not None and self.coord.shape[-1] != 2:
            raise ValueError('NumPy data conversion requires 2D coordinates.')
        values = {'coord': self.coord, 'disp': self.disp, 'hd': self.hd}
        for name, value in values.items():
            if value is not None:
                value = value.detach()
                value = value.cpu()
                values[name] = value.numpy()
        return NumPyTrajectory(**values)

    def reshape(self, shape):
        if len(shape) != 2:
            raise ValueError('Shape must contain batch and time dimensions.')
        for name in self.fields:
            value = getattr(self, name)
            if value is not None:
                value = value.reshape(*shape, value.shape[-1])
                setattr(self, name, value)
        return self

    def t_range(self, range_):
        start, stop = range_
        if not 0 <= start < stop <= self.n_steps:
            raise ValueError('Invalid time range.')
        return self[:, start:stop]

    @classmethod
    def from_numpy(cls, traj, device='cpu'):
        """Adapt the original NumPy motion output to the tensor interface."""
        coord = torch.as_tensor(traj.coord, dtype=torch.float32, device=device)
        disp = torch.as_tensor(traj.disp, dtype=torch.float32, device=device)
        spd = disp.norm(dim=-1, keepdim=True)
        denominator = spd.clamp_min(1e-12)
        mv_dir = disp / denominator
        angle = torch.as_tensor(traj.hd, dtype=torch.float32, device=device)
        angle = angle.squeeze(-1)
        row = torch.sin(angle)
        column = torch.cos(angle)
        head_dir = torch.stack([row, column], dim=-1)
        return cls(coord=coord, head_dir=head_dir, spd=spd, mv_dir=mv_dir, device=device)

    def state_dict(self):
        return {name: getattr(self, name) for name in self.fields}

    @classmethod
    def from_dict(cls, state_dict):
        coord = state_dict['coord']
        return cls(**state_dict, device=coord.device)

    @staticmethod
    def load(path):
        raise ValueError('Use Trajectory.from_dict with tensor fields.')
