"""The original NumPy random walk behind the standard control interface."""

from .autonomous_behavior import AutonomousBehavior
from .controllable_behavior import ControllableBehavior
from .behavior_config import BehaviorConfig
from rtgym.dataclasses import Trajectory


class Behavior:
    def __init__(self, gym):
        self.gym = gym
        self.device = gym.device
        self.config = BehaviorConfig()
        self.config.register_callback(self._on_config_change)
        self.autonomous = AutonomousBehavior(gym, self.config)
        self.controllable = ControllableBehavior(gym, self.config)
        self.cur_state = None
        self.raw_state = None

    def init_from_profile(self, profile):
        if self.gym.arena.ndim != 2:
            raise ValueError('random_walk requires a 2D arena.')
        self.autonomous.init_from_profile(profile)
        self.controllable.init_from_profile(profile)

    def _on_arena_change(self):
        self.autonomous._recompute_maps()
        self.controllable._recompute_maps()
        self.reset()

    def generate_trajectory(self, duration_ts, batch_size, init_state=None):
        # Retain the full NumPy generator state during consecutive calls.
        raw_state = None
        init_pos = None
        if init_state is self.cur_state and self.raw_state is not None:
            raw_state = self.raw_state
        elif init_state is not None and init_state.coord is not None:
            init_pos = init_state.coord.detach()
            init_pos = init_pos.cpu()
            init_pos = init_pos.numpy()
        duration = self.gym.to_sec(duration_ts)
        traj, raw_state = self.autonomous.generate_trajectory(duration, batch_size, init_pos, raw_state)
        self.raw_state = raw_state
        traj = Trajectory.from_numpy(traj, device=self.device)
        self.cur_state = traj[:, -1]
        return traj, self.cur_state

    def reset(self):
        self.cur_state = None
        self.raw_state = None
        self.controllable.reset()

    def step(self, state, mv_dir, spd, head_dir):
        raise ValueError('Use trajectory_generator control for vector-command stepping.')

    def _on_config_change(self, attr, value):
        if attr == 'avoid_boundary_dist':
            self._on_arena_change()
