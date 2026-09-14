# behavior.py
from .autonomous_behavior import AutonomousBehavior
from .controllable_behavior import ControllableBehavior
from .behavior_config import BehaviorConfig
from .ema_behavior import EMABehavior


class Behavior:
    def __init__(self, gym):
        self.gym = gym
        self.ema = None
        self.config = BehaviorConfig()
        self.config.register_callback(self._on_config_change)
        self.autonomous = AutonomousBehavior(gym, self.config)
        self.controllable = ControllableBehavior(gym, self.config)
    
    def init_from_profile(self, raw_profile):
        if raw_profile.get('type') == 'ema':
            self.ema = EMABehavior(self.gym, raw_profile)
            return
        if self.gym.arena.ndim != 2:
            raise ValueError('The original behavior requires a 2D arena; use type=ema for 3D.')
        self.ema = None
        self.autonomous.init_from_profile(raw_profile)
        self.controllable.init_from_profile(raw_profile)
    
    def _on_arena_change(self):
        if self.ema is not None:
            self.ema._recompute_maps()
            self.ema.cur_state = None
            return
        if self.gym.arena.ndim != 2:
            return
        self.autonomous._recompute_maps()
        self.controllable._recompute_maps()
    
    def generate_trajectory(self, duration: float, batch_size: int, init_pos=None, init_state=None):
        if self.ema is not None:
            n_steps = self.gym.to_ts(duration)
            return self.ema.generate_steps(n_steps, batch_size, init_state, init_pos)
        return self.autonomous.generate_trajectory(duration, batch_size, init_pos, init_state)

    def _on_config_change(self, attr, value):
        if attr == "avoid_boundary_dist":
            self._on_arena_change()
