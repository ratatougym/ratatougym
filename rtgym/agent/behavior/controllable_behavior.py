import numpy as np
from .base_behavior import BaseBehavior
from rtgym.dataclass import RawTrajectory


class ControllableBehavior(BaseBehavior):
    """
    The controller module contains the Controller class which is responsible for 
    controlling the agent's behavior.
    """
    def __init__(self, gym, config):
        super().__init__(gym, config)
        self.traj = RawTrajectory()

    def init_from_profile(self, raw_profile):
        required_keys = ['max_velocity', 'avoid_boundary_dist']
        if all(key in raw_profile for key in required_keys):
            self.config.max_velocity = raw_profile['max_velocity']
            self.config.avoid_boundary_dist = raw_profile['avoid_boundary_dist']
            self._recompute_maps()
            self.initialized = True
            print("Controllable behavior initialized")
        else:
            self.initialized = False
            missing = [key for key in required_keys if key not in raw_profile]
            print(f"Controllable behavior not initialized: missing keys {missing}")

    def step(self, agent_state, displacement):
        agent_state.disp = displacement.copy()
        agent_state.vel_norm = np.linalg.norm(displacement, axis=-1)
        agent_state.mv_dir = np.arctan2(displacement[..., 1], displacement[..., 0]).reshape(-1, 1)
        self._clip_displacement(agent_state.disp)
        self._update_coord(agent_state)  # Update the coordinate of the agent and avoid boundary
        self.traj.append(agent_state)
        return agent_state
    
    def _clip_displacement(self, displacement):
        """
        For all those displacements that are larger than self.config.max_velocity, 
        we will normalize them to self.config.max_velocity.
        """
        displacement_length = np.linalg.norm(displacement, axis=1) / self.gym.t_res * 1e3
        mask = displacement_length > self.config.max_velocity
        if np.any(mask):
            displacement[mask] = displacement[mask] / displacement_length[mask].reshape(-1, 1) * self.config.max_velocity

    def get_trajectory(self):
        return self.traj.to_trajectory()

    def reset(self):
        self.traj = RawTrajectory()
