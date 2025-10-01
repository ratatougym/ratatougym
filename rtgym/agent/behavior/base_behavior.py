import numpy as np
from scipy.ndimage import distance_transform_edt
from functools import wraps
from .behavior_config import BehaviorConfig


class BaseBehavior:
    """
    Base class for agent behavior systems.

    Note: Not supposed to be used directly.
    
    Provides common functionality for behavior generation including boundary
    avoidance, coordinate updates, and movement computations. This is an
    abstract base class that should be inherited by specific behavior types.
    
    Args:
        gym (RatatouGym): Parent gym environment.
        config (BehaviorConfig): Configuration object for behavior parameters.
        
    Attributes:
        gym (RatatouGym): Parent gym environment.
        config (BehaviorConfig): Behavior configuration.
        initialized (bool): Whether behavior has been initialized.
        distance_map (np.ndarray): Distance map for boundary avoidance.
        perpend_angle_map (np.ndarray): Perpendicular angle map for boundaries.
    """
    
    initialized = False
    
    @staticmethod
    def require_init(func):
        """Decorator to ensure behavior is initialized before method execution.
        
        Args:
            func (callable): Function to wrap with initialization check.
            
        Returns:
            callable: Wrapped function that checks initialization.
            
        Raises:
            ValueError: If behavior is not initialized.
        """
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            if not instance.initialized:
                raise ValueError("Behavior not initialized.")
            return func(instance, *args, **kwargs)
        return wrapper

    def __init__(self, gym, config):
        self.gym = gym
        self.config = config

    @property
    def t_res(self):
        """Get temporal resolution from gym.
        
        Returns:
            float: Temporal resolution in milliseconds.
        """
        return self.gym.t_res

    @property
    def s_res(self):
        """Get spatial resolution from gym.
        
        Returns:
            float: Spatial resolution in units per pixel.
        """
        return self.gym.s_res

    @property
    def arena(self):
        """Get arena from gym.
        
        Returns:
            Arena: The arena environment.
        """
        return self.gym.arena

    def _recompute_maps(self):
        if self.config.avoid_boundary_dist <= 0:
            return
        # Compute the distance map: Euclidean distance to the nearest boundary (obstacle)
        raw_distance_map = distance_transform_edt(1 - self.arena.arena_map)

        # Apply Gaussian-like weighting based on the width
        distance_map = np.exp(- (raw_distance_map**2 / self.config.avoid_boundary_dist))  # Smoothly decays to zero beyond the width

        # Compute the gradient of the raw distance map
        gradient_y, gradient_x = np.gradient(raw_distance_map)

        # Compute the angle of the gradient at each point and normalize to [0, pi)
        perpend_angle_map = np.arctan2(-gradient_y, gradient_x)  # Flip the y-axis to match the coordinate system
        # Rotate left by 90 degrees to align with the direction of the gradient
        perpend_angle_map = np.mod(perpend_angle_map + np.pi/2, 2*np.pi) - np.pi

        self.distance_map = distance_map
        self.perpend_angle_map = perpend_angle_map

    @staticmethod
    def _flip_coin(p, size):
        return np.random.choice([False, True], p=[1-p, p], size=size)

    @staticmethod
    def _random_rad(size):
        """Generate random direction vectors in radians.
        
        Creates random directional vectors uniformly distributed between -π and π.
        
        Args:
            size (tuple): Shape of the output array.
            
        Returns:
            np.ndarray: Direction vector of shape (*size, 1) with values in [-π, π).
        """
        return np.random.uniform(-np.pi, np.pi, size=size)[..., np.newaxis]
    
    @staticmethod
    def _rad_vel_to_disp(direction, velocity_norm, scale=1, drift=None):
        """Scale direction vectors by velocity norm.

        Args:
            direction: Direction vector. np.ndarray of shape (batch_size, 1).
            velocity_norm: Velocity norm of the agent. np.ndarray of shape (batch_size, 1).
            scale: Scale factor for the displacement. Default is 1.
            drift: Optional drift to add to displacement.
        """
        # Scale by velocity norms
        direction = np.column_stack((np.cos(direction), np.sin(direction)))
        displacement = direction * velocity_norm[:, np.newaxis] * scale
        if drift is not None:
            displacement += drift
        return displacement

    @staticmethod
    def _compute_direction(displacement):
        """Compute the direction of the displacement in radians.

        Args:
            displacement: Displacement of the agent. np.ndarray of shape (batch_size, 2).
        """
        direction = np.arctan2(displacement[..., 1], displacement[..., 0])
        return direction[..., np.newaxis]

    def _update_coord(self, raw_agent_state):
        """Update the coordinate of the agent.

        Args:
            raw_agent_state: Generator state of the agent. A rtgym.dataclass.GeneratorState object.
        """
        # First, try to move the agent and see if it hits the boundary
        updated_coord = raw_agent_state.coord + raw_agent_state.disp
        invalid_batches = np.logical_not(self.arena.validate_index(updated_coord.astype(int)))
        while np.sum(invalid_batches) > 0:
            # Generate new movement direction for those batches that hit the wall
            raw_agent_state.mv_dir[invalid_batches] = self._random_rad(np.sum(invalid_batches))

            # Generate new displacement
            disp = self._rad_vel_to_disp(
                direction=raw_agent_state.mv_dir[invalid_batches], 
                velocity_norm=raw_agent_state.vel_norm[invalid_batches]
            )

            # Update the location of the batches that hit the wall and check again
            updated_coord[invalid_batches] = raw_agent_state.coord[invalid_batches] + disp
            invalid_batches = np.logical_not(self.arena.validate_index(updated_coord.astype(int)))

        # If not, update the location of the agent
        raw_agent_state.coord = updated_coord
