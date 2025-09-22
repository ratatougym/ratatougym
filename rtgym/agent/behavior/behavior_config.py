from dataclasses import dataclass


@dataclass
class BehaviorConfig:
    def __init__(self):
        self._velocity_mean = 0.0
        self._velocity_sd = 0.0
        self._random_drift_magnitude = 0.0
        self._switch_direction_prob = 0.0
        self._switch_velocity_prob = 0.0
        self._max_velocity = 0.0
        self._avoid_boundary_dist = -1
        self._callbacks = []

    def register_callback(self, callback):
        self._callbacks.append(callback)

    def _notify(self, attr, value):
        for callback in self._callbacks:
            callback(attr, value)

    @property
    def avoid_boundary_dist(self):
        return self._avoid_boundary_dist

    @avoid_boundary_dist.setter
    def avoid_boundary_dist(self, value):
        self._avoid_boundary_dist = value
        self._notify("avoid_boundary_dist", value)
