# behavior.py
from .autonomous_behavior import AutonomousBehavior
from .controllable_behavior import ControllableBehavior
from .behavior_config import BehaviorConfig


class Behavior:
    def __init__(self, gym):
        self.config = BehaviorConfig()
        self.config.register_callback(self._on_config_change)
        self.autonomous = AutonomousBehavior(gym, self.config)
        self.controllable = ControllableBehavior(gym, self.config)
    
    def init_from_profile(self, raw_profile):
        self.autonomous.init_from_profile(raw_profile)
        self.controllable.init_from_profile(raw_profile)
    
    def _on_arena_change(self):
        self.autonomous._recompute_maps()
        self.controllable._recompute_maps()
    
    def generate_trajectory(self, duration: float, batch_size: int, init_pos=None, init_state=None):
        return self.autonomous.generate_trajectory(duration, batch_size, init_pos, init_state)

    def _on_config_change(self, attr, value):
        if attr == "avoid_boundary_dist":
            self._on_arena_change()
