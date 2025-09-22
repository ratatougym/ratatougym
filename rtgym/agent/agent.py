import numpy as np
from rtgym.agent.behavior import Behavior
from rtgym.agent.sensory import Sensory
from rtgym.dataclass import AgentState, Trajectory, RawAgentState, RawTrajectory
from typing import Union


class Agent():
    """
    The class object of the Agent. An agent represents the subject that navigates 
    or traverses the arena, either randomly or under control. This implementation 
    manages two central aspects: how the agent generates trajectories through the 
    environment, and how different sensory systems or spatial navigation cell types 
    might respond along those trajectories.

    The Agent can be thought of as a virtual animal whose movement and sensory 
    inputs can be precisely controlled and observed. In this setting, theoretical 
    models of spatially tuned cells—such as place cells, grid cells, or head 
    direction cells—can be applied to predict how each of them would respond given 
    the positions visited along a trajectory.

    The Agent acts as a unifying interface that combines behavioral rules with 
    sensory transformations, enabling reproducible experiments where navigation 
    and perception are directly linked.

    The Agent is coupled to a gym environment that defines the arena, spatial and 
    temporal resolution, and the overall rules of the task. Through this interface, 
    the Agent can be spawned at specific or random locations, controlled step by 
    step, or allowed to move autonomously, while its sensory responses can be 
    queried from individual states or full trajectories.
    
    Args:
        gym (RatatouGym): The parent gym environment.
        behavior_profile (dict): Current behavior configuration.
        sensory_profile (dict): Current sensory configuration.
        sensory (Sensory): Sensory system manager.
        behavior (Behavior): Behavior system manager.
    """
    
    def __init__(self, gym):
        self.gym = gym
        self.behavior_profile = None
        self.sensory_profile = None
        self.sensory = Sensory(self.gym)
        self.behavior = Behavior(self.gym)
        self._state = RawAgentState()

    @property
    def temporal_resolution(self):
        """Get temporal resolution from gym.
        
        Returns:
            float: Temporal resolution in milliseconds.
        """
        return self.gym.temporal_resolution

    @property
    def spatial_resolution(self):
        """Get spatial resolution from gym.
        
        Returns:
            float: Spatial resolution in units per pixel.
        """
        return self.gym.spatial_resolution

    @property
    def sensories(self):
        """Get all sensory modalities.
        
        Returns:
            dict: Dictionary of sensory modalities keyed by name.
        """
        return self.sensory.sensories

    @property
    def arena(self):
        """Get arena from gym.
        
        Returns:
            Arena: The arena environment.
        """
        return self.gym.arena
    
    @property
    def controllable(self):
        """Get controllable behavior system.
        
        Returns:
            ControllableBehavior: Controllable behavior manager.
        """
        return self.behavior.controllable

    @property
    def autonomous(self):
        """Get autonomous behavior system.
        
        Returns:
            AutonomousBehavior: Autonomous behavior manager.
        """
        return self.behavior.autonomous

    @property
    def state(self):
        """Get current agent state.
        
        Returns:
            RawAgentState: Current raw agent state.
        """
        return self._state

    def _on_arena_change(self):
        """Handle arena change events.
        
        Updates behavior and sensory systems when the arena changes.
        """
        if self.arena is not None:
            self.behavior._on_arena_change()
            self.sensory._on_arena_change()
            self._init_behavior_from_profile()
            self._init_sensory_from_profile()

    def set_behavior(self, behavior_profile):
        """Set behavior configuration.
        
        Args:
            behavior_profile (dict): Behavior configuration parameters.
        """
        self.behavior_profile = behavior_profile
        self._init_behavior_from_profile()

    def _init_behavior_from_profile(self):
        """Initialize behavior from behavior profile.
        
        Sets up the behavior system using the current behavior profile
        if both profile and arena are available.
        """
        if self.behavior_profile is not None and self.arena is not None:
            self.behavior.init_from_profile(self.behavior_profile)

    def set_sensory(self, sensory_profile):
        """Set sensory configuration.
        
        Args:
            sensory_profile (dict): Sensory configuration parameters.
        """
        self.sensory_profile = sensory_profile
        self._init_sensory_from_profile()

    def add_sensory(self, sensory_profile):
        """Add sensory modalities to existing configuration.
        
        Args:
            sensory_profile (dict): Additional sensory configuration parameters.
        """
        self.sensory_profile.update(sensory_profile)
        self.sensory.add_sensory(sensory_profile)

    def set_sensory_manually(self, sens_type, sens):
        import warnings
        warnings.warn("DeprecationWarning: The method 'set_sensory_manually' is deprecated. "
                      "Potential implementation issues might cause unintended behavior. "
                      "It is recommended to implement a separate sensory class instead.")
        # if self.sensory is not None:
        #     print('Warning: sensory is reset')
        # self.sensory.set_sensory_manually(sens_type, sens)
        # self.sensory = sensory

    @property
    def sensories(self):
        """Get all sensory modalities.
        
        Returns:
            dict: Dictionary of sensory modalities keyed by name.
        """
        return self.sensory.sensories

    def _init_sensory_from_profile(self):
        """
        Initialize sensory from sensory profile
        """
        if self.sensory_profile is not None and self.arena is not None:
            self.sensory.init_from_profile(self.sensory_profile)

    # ================================
    # Behavior
    # ================================
    def random_traverse(self, duration: float, batch_size: int, init_pos=None, init_state=None, pause_prob=0):
        traj, state = self.behavior.generate_trajectory(duration, batch_size, init_pos, init_state)
        if pause_prob > 0:
            pause_mask = np.random.rand(batch_size) < pause_prob
            traj.disp[pause_mask] = np.zeros_like(traj.disp[pause_mask])
            traj.coord[pause_mask] = traj.coord[pause_mask, 0][:, np.newaxis]
            traj.hd[pause_mask] = np.zeros_like(traj.hd[pause_mask])
            # print("Continuing trajectory maybe deprecated when pause_prob > 0")
        self._state = state
        return traj

    def step(self, displacement):
        self._state = self.controllable.step(self._state, displacement)

    def get_response(
            self,
            agent_data: Union[AgentState, Trajectory],
            return_format='array', 
            keys=None, 
            str_filter=None, 
            type_filter=None
        ):
        if isinstance(agent_data, RawAgentState):
            agent_data = agent_data.to_agent_state()
        elif isinstance(agent_data, RawTrajectory):
            agent_data = agent_data.to_trajectory()
        return self.sensory.get_response(agent_data, return_format, keys, str_filter, type_filter)

    def spawn(self, init_pos=None, init_state=None):
        """Spawn the controllable agent at the given position and state.
        
        Resets the controllable behavior and initializes the agent at a specific
        location and state for manual control.
        
        Args:
            init_pos (np.ndarray, optional): Initial position of the agent.
            init_state (AgentState or RawAgentState, optional): Initial state of the agent. 
                If both init_pos and init_state are provided, init_state will be used.
        """
        self.controllable.reset()
        if init_state is not None:
            if isinstance(init_state, RawAgentState):
                self._state = init_state.copy()
            elif isinstance(init_state, AgentState):
                self._state.reset()
                self._state.coord = init_state.coord
                self._state.disp = init_state.disp
        elif init_pos is not None:
            self._state.reset()
            self._state.coord = init_pos

    def random_spawn(self, batch_size: int):
        """
        Spawn the controller at a random position in the arena.
        """
        self.spawn(init_pos=self.arena.random_position(batch_size))

    def step(self, displacement):
        self._state = self.controllable.step(self._state, displacement)
