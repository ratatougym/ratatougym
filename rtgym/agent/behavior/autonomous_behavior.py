import numpy as np
from scipy.ndimage import gaussian_filter1d
from .base_behavior import BaseBehavior
from rtgym.dataclass import RawTrajectory, RawAgentState


class AutonomousBehavior(BaseBehavior):
    """
    The AutonomousBehavior class is responsible for generating random traversal
    trajectories for the agent.

    This class implements autonomous agent behavior that generates trajectories 
    through random walks (with momentum). It generates trajectories with 
    configurable parameters including velocity distributions, random drift, 
    direction switching probabilities, and boundary avoidance.
    """
    def __init__(self, gym, config):
        super().__init__(gym, config)
        self.raw_agent_state = RawAgentState()

    def _check_params(self):
        """
        Validate behavior configuration parameters.

        Ensures all configuration parameters are within valid ranges and meet
        the requirements for trajectory generation.

        Raises:
            AssertionError: If any parameter is outside valid range.
        """
        assert self.config.velocity_mean > 0, 'velocity_mean must be positive'
        assert self.config.random_drift_magnitude >= 0, 'random_drift_magnitude must be non-negative'
        assert 0 <= self.config.switch_direction_prob <= 1, 'switch_direction_prob must be in [0, 1]'
        assert 0 <= self.config.switch_velocity_prob <= 1, 'switch_velocity_prob must be in [0, 1]'

    def init_from_profile(self, raw_profile):
        """
        Initialize behavior configuration from a profile dictionary.

        Extracts and validates behavior parameters from the provided profile.
        Sets default values for optional parameters and checks parameter validity.

        Args:
            raw_profile (dict): Configuration dictionary containing behavior parameters.
                Required keys: 
                    - velocity_mean
                    - random_drift_magnitude
                    - switch_direction_prob
                    - switch_velocity_prob
                Optional keys: 
                    - velocity_sd
                    - avoid_boundary_dist
        """
        required_keys = ['velocity_mean', 'random_drift_magnitude', 'switch_direction_prob', 'switch_velocity_prob']
        if all(key in raw_profile for key in required_keys):
            self.config.velocity_mean = raw_profile['velocity_mean']
            self.config.velocity_sd = max(raw_profile.get('velocity_sd', 0), 0)
            self.config.random_drift_magnitude = raw_profile['random_drift_magnitude']
            self.config.switch_direction_prob = raw_profile['switch_direction_prob']
            self.config.switch_velocity_prob = raw_profile['switch_velocity_prob']
            self.config.avoid_boundary_dist = raw_profile.get('avoid_boundary_dist', -1)
            self._check_params()
            self._recompute_maps()
            self.initialized = True
            print("Autonomous behavior initialized")
        else:
            self.initialized = False
            missing = [key for key in required_keys if key not in raw_profile]
            print(f"Autonomous behavior not initialized: missing keys {missing}")

    @BaseBehavior.require_init
    def generate_trajectory(self, duration: float, batch_size: int, init_pos=None, init_state=None):
        """
        Generate a trajectory and return generated data.
        
        Creates autonomous agent trajectories based on the configured behavior
        parameters including random walks and boundary avoidance. The trajectory
        generation process involves initializing agent states, generating movement
        at each timestep, computing displacements, and calculating head directions.
        
        Args:
            duration (float): Duration of the trial in seconds.
            batch_size (int): Number of trials to generate.
            init_pos (np.ndarray, optional): Initial position of the agent.
            init_state (AgentState, optional): Initial state of the agent. 
                If both init_pos and init_state are provided, init_state will be used.
                
        Returns:
            tuple: A tuple containing:
                - Trajectory: Generated trajectory as a rtgym.dataclass.Trajectory object.
                - RawAgentState: Final state of the agent after trajectory generation.
        """
        self.dur_ts = self.gym.to_ts(duration)
        self.batch_size = batch_size

        # Initialize the raw trajectory
        raw_traj = RawTrajectory(
            dur_ts=self.dur_ts,
            batch_size=self.batch_size
        )
        self.raw_agent_state = self._init_raw_agent_state(init_pos=init_pos, init_state=init_state)
        raw_traj.update_state(0, self.raw_agent_state)

        # Generate the trajectory
        self._generate_trajectory(raw_traj)
        self._compute_displacements(raw_traj)
        self._generate_hds(raw_traj)

        return raw_traj.to_trajectory(), self.raw_agent_state

    def _random_velocity(self, size):
        """Generate random velocity values from a log-normal distribution.
        
        Converts the configured mean and standard deviation parameters into
        the underlying Gaussian distribution parameters for log-normal sampling.
        Scales the results by temporal resolution.
        
        Args:
            size (int or tuple): Shape of the output array.
            
        Returns:
            np.ndarray: Random velocity values scaled for the simulation timestep.
        """
        # Solve for the mean and sigma of the underlying Gaussian
        # distribution with which we used to generate log-normal
        m = self.config.velocity_mean
        s = self.config.velocity_sd
        sigma = np.sqrt(np.log(1 + (s**2 / m**2)))
        mu = np.log(m**2 / np.sqrt(m**2 + s**2))
        dist = np.random.lognormal(mean=mu, sigma=sigma, size=size)
        return dist / 1e3 * self.t_res

    def _compute_displacements(self, raw_traj):
        """Compute displacement vectors from coordinate differences.
        
        Calculates displacement vectors between consecutive positions and
        pads with zeros at the end to maintain consistent array dimensions.
        
        Args:
            raw_traj (RawTrajectory): Raw trajectory object to update with displacements.
        """
        raw_traj.displacements = np.diff(raw_traj.coord, axis=1)
        raw_traj.displacements = np.concatenate([
            raw_traj.displacements, 
            np.zeros((raw_traj.batch_size, 1, 2))
        ], axis=1)

    def _generate_hds(self, raw_traj):
        """Generate head directions based on movement direction.
        
        Computes head direction as the angle of movement direction,
        aligning the agent's heading with its movement vector using
        arctan2 to handle all quadrants correctly.
        
        Args:
            raw_traj (RawTrajectory): Raw trajectory object to update with head directions.
        """
        # # TODO: Change scale
        # raw_traj.hd[:] = np.random.normal(loc=0, scale=0.2, size=raw_traj.hd.shape)
        # raw_traj.hd[:] = gaussian_filter1d(raw_traj.hd, sigma=5, axis=1)

        # # Add the angle of the movement to the head direction
        # angle_rad = np.arctan2(raw_traj.displacements[..., 0], raw_traj.disp[..., 1])
        # raw_traj.hd += angle_rad[..., np.newaxis]
        angle_rad = np.arctan2(raw_traj.displacements[..., 0], raw_traj.disp[..., 1])
        raw_traj.hd[:] = angle_rad[..., np.newaxis]

    def _generate_trajectory(self, raw_traj):
        """Generate the main trajectory by iterating through timesteps.
        
        For each timestep, updates agent coordinates, adjusts drifting direction
        based on switching probabilities, updates velocity, computes new displacement
        vectors, applies boundary avoidance, and stores the state in the trajectory.
        
        Args:
            raw_traj (RawTrajectory): Raw trajectory object to populate with generated data.
        """
        for ts in range(1, self.dur_ts):
            self._update_coord(self.raw_agent_state)
            self._update_drifting_direction(self.raw_agent_state)
            self._update_velocity(self.raw_agent_state)

            # Update the displacement, direction, and velocity
            self.raw_agent_state.disp = self._rad_vel_to_disp(
                direction=self.raw_agent_state.mv_dir, 
                velocity_norm=self.raw_agent_state.vel_norm,
                drift=self.raw_agent_state.drift
            )
            self.raw_agent_state.mv_dir = self._avoid_boundary(
                direction=self._compute_direction(self.raw_agent_state.disp),
                coords=self.raw_agent_state.coord
            )
            raw_traj.update_state(ts, self.raw_agent_state)
    
    def _update_drifting_direction(self, raw_agent_state):
        """Update the drifting direction of the agent based on switching probability.
        
        Randomly determines which agents should switch their drift direction
        based on the configured switch_direction_prob. For agents that switch,
        generates new random drift vectors with the configured magnitude.

        Args:
            raw_agent_state (RawAgentState): Generator state of the agent containing
                current drift, velocity, and batch information.
        """
        # Randomly switch the direction of the random drift
        switch_direction_idx = self._flip_coin(self.config.switch_direction_prob, raw_agent_state.batch_size)
        n_switch = np.sum(switch_direction_idx)
        if n_switch > 0:
            raw_agent_state.drift[switch_direction_idx] = self._rad_vel_to_disp(
                direction=self._random_rad(n_switch),
                velocity_norm=raw_agent_state.vel_norm[switch_direction_idx],
                scale=self.config.random_drift_magnitude
            )

    def _update_velocity(self, raw_agent_state):
        """Update the velocity of the agent based on switching probability.
        
        Randomly determines which agents should change their velocity based on
        the configured switch_velocity_prob. For agents that switch, samples
        new velocity values from the configured distribution.

        Args:
            raw_agent_state (RawAgentState): Generator state of the agent containing
                current velocity norms and batch information.
        """
        # Randomly switch the velocity
        switch_velocity_idx = self._flip_coin(self.config.switch_velocity_prob, raw_agent_state.batch_size)
        n_switch = np.sum(switch_velocity_idx)
        if n_switch > 0:
            raw_agent_state.vel_norm[switch_velocity_idx] = self._random_velocity(n_switch)

    def _compute_boundary_avoidance_adjustment(self, direction, perpend_angle):
        """Compute the boundary avoidance adjustment for the agent.
        
        Determines whether the agent is pointing toward a boundary and calculates
        the complement angle needed to redirect the agent away from the boundary.
        Uses the perpendicular angle to the nearest boundary to determine the
        appropriate avoidance direction.

        Args:
            direction (np.ndarray): Current direction of the agent with shape (batch_size, 1).
            perpend_angle (np.ndarray): Perpendicular angle to the nearest boundary.
                This determines which side the agent should drift away from to avoid
                sharp turns when near boundaries.
                
        Returns:
            tuple: A tuple containing:
                - pointing_boundary (np.ndarray): Binary array indicating if agent
                  is pointing toward boundary (1) or away (0).
                - complement_angle (np.ndarray): Angle adjustment needed to redirect
                  agent away from boundary.
        """
        # Compute angle difference
        angle_diff = perpend_angle[..., np.newaxis] - direction
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi)

        # Determine pointing boundary
        pointing_boundary = (angle_diff >= -np.pi / 2) & (angle_diff <= np.pi / 2)
        pointing_boundary = pointing_boundary.astype(int)

        # Compute complement angle
        complement_angle = -np.sign(angle_diff) * (np.pi / 2 - np.abs(angle_diff))

        return pointing_boundary, complement_angle

    def _init_raw_agent_state(self, init_pos=None, init_state=None):
        """Initialize the generator state of the agent.
        
        Creates the initial state for trajectory generation. If initial position
        or state is provided, uses that as starting point. Otherwise, places
        agents at random valid positions and initializes velocity and drift
        parameters according to the configured distributions.
        
        The generator state aligns displacement and coordinates such that
        coordinates represent current position and displacement represents
        the movement that will occur in the next timestep.

        Args:
            init_pos (np.ndarray, optional): Initial position with shape (batch_size, 2).
            init_state (RawAgentState, optional): Complete initial state object.
                If provided, takes precedence over init_pos.
        
        Returns:
            RawAgentState: Initialized generator state containing coordinates,
                velocity norms, drift vectors, displacement vectors, and movement
                direction with boundary avoidance applied.
        """
        raw_agent_state = RawAgentState()   
        if init_state is not None:
            raw_agent_state = init_state
        else:
            if init_pos is not None:
                raw_agent_state.coord = init_pos
            else:
                raw_agent_state.coord = self.arena.generate_random_pos(self.batch_size)
            raw_agent_state.vel_norm = self._random_velocity(self.batch_size)
            raw_agent_state.drift = self._rad_vel_to_disp(
                direction=self._random_rad(self.batch_size), 
                velocity_norm=raw_agent_state.vel_norm,
                scale=self.config.random_drift_magnitude
            )
            raw_agent_state.disp = self._rad_vel_to_disp(
                direction=self._random_rad(self.batch_size), 
                velocity_norm=raw_agent_state.vel_norm,
                drift=raw_agent_state.drift
            )
            gt_direction = self._compute_direction(raw_agent_state.disp)
            raw_agent_state.mv_dir = self._avoid_boundary(gt_direction, raw_agent_state.coord)

        return raw_agent_state
    
    def _avoid_boundary(self, direction, coords):
        """Adjust agent direction to avoid arena boundaries.
        
        Applies a "force" to push agents away from boundaries when they are
        within the configured avoidance distance. This is a predictive avoidance
        mechanism that prevents agents from getting too close to walls, separate
        from collision detection in coordinate updates.
        
        Uses precomputed distance and angle maps to determine boundary proximity
        and calculates appropriate direction adjustments to smoothly guide
        agents away from boundaries.

        Args:
            direction (np.ndarray): Current direction of agents with shape (batch_size, 1).
            coords (np.ndarray): Current coordinates of agents with shape (batch_size, 2).
            
        Returns:
            np.ndarray: Adjusted directions with boundary avoidance applied.
                Agents pointing away from boundaries are unchanged, while those
                pointing toward nearby boundaries have their direction modified.
        """
        if self.config.avoid_boundary_dist <= 0:
            return direction
        int_coords = coords.astype(int)
        avoid_coef = self.distance_map[int_coords[:, 0], int_coords[:, 1]]
        avoid_idx = np.where(avoid_coef > 1e-2)[0]  # Batches that are close to the boundary

        if len(avoid_idx) > 0:
            # If there are batches that are close to the boundary, adjust the direction
            perpend_angle = self.perpend_angle_map[int_coords[avoid_idx, 0], int_coords[avoid_idx, 1]]
            pointing_boundary, complement_angle = self._compute_boundary_avoidance_adjustment(
                direction[avoid_idx], perpend_angle
            )
            
            # Compute the adjustment coefficient
            avoid_coef = avoid_coef[avoid_idx][..., np.newaxis]
            avoid_coef = avoid_coef * pointing_boundary  # Not adjust if pointing away from the boundary

            # Adjust the direction
            adjusted_direction = (1 - avoid_coef) * direction[avoid_idx]
            boundary_adjustment = avoid_coef * (direction[avoid_idx] + complement_angle)
            direction[avoid_idx] = adjusted_direction + boundary_adjustment

        return direction
