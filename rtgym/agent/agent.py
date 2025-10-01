import numpy as np
from rtgym.agent.behavior import Behavior
from rtgym.agent.neurons import Neurons
from rtgym.dataclass import AgentState, Trajectory, RawAgentState, RawTrajectory
from typing import Union


class Agent():
    """
    The Agent represents a virtual subject that navigates within an arena and 
    generates neuronal responses based on its movement. It serves as the 
    encapsulating object for two main components:

    - **Behavior system**: Governs how the agent moves, either autonomously 
        (random or rule-based trajectories) or under manual control.
    - **Neurons**: Simulates neuronal groups (e.g., place cells, grid cells, head 
        direction cells) that respond to the agent's movement and position.

    The Agent is coupled to a gym environment that defines the arena, as well as 
    the spatial and temporal resolution of the simulation. It can be spawned at 
    chosen or random positions, moved step by step, or allowed to traverse 
    autonomously. During these interactions, its neuronal responses can be 
    queried for individual states or full trajectories.

    Conceptually, the Agent can be regarded as a virtual animal whose navigation 
    and neuronal responses are directly linked, providing a reproducible interface 
    for testing theoretical models of spatially tuned cells.

    Args:
        gym (RatatouGym): Parent gym environment providing the arena, resolution, 
            and task rules.
    """
    
    def __init__(self, gym):
        """Initialize an Agent instance.
        
        Args:
            gym (RatatouGym): The parent gym environment that provides the arena,
                temporal/spatial resolution, and task rules.
        """
        self.gym = gym
        self.behavior_profile = None
        self.neuron_profiles = None
        self.neurons = Neurons(self.gym)
        self.behavior = Behavior(self.gym)
        self._state = RawAgentState()

    @property
    def neuron_groups(self):
        """
        Get all neuron groups of the agent.
        
        Returns:
            dict: Dictionary of neuron groups keyed by name.
        """
        return self.neurons.neuron_groups

    @property
    def arena(self):
        """
        Point to the arena environment.
        
        Returns:
            Arena: The arena environment.
        """
        return self.gym.arena

    @property
    def autonomous(self):
        """
        Get autonomous behavior system.
        
        Returns:
            AutonomousBehavior: Autonomous behavior manager.
        """
        return self.behavior.autonomous

    @property
    def state(self):
        """Get current agent state.
        
        Returns:
            RawAgentState: The internal state used to keep the agent's movement 
                during trajectory generation.
        """
        return self._state

    def _on_arena_change(self):
        """
        Handle arena change events.
        
        Updates behavior and neurons when the arena changes.
        """
        if self.arena is not None:
            self.behavior._on_arena_change()
            self.neurons._on_arena_change()
            self._init_behavior_from_profile()
            self._init_neurons_from_profile()

    def init_behavior(self, behavior_profile: dict):
        """
        Initialize the agent's behavior.

        Args:
            behavior_profile (dict): Behavior configuration parameters.
        """
        self.behavior_profile = behavior_profile
        self._init_behavior_from_profile()

    def _init_behavior_from_profile(self):
        """
        Initialize behavior from behavior profile.
        
        Sets up the behavior system using the current behavior profile
        if both profile and arena are available.
        """
        if self.behavior_profile is not None and self.arena is not None:
            self.behavior.init_from_profile(self.behavior_profile)

    def init_neurons(self, neuron_profiles: dict):
        """
        Initialize the agent's neurons.

        Args:
            neuron_profile (dict): Neurons configuration parameters.
        """
        self.neuron_profiles = neuron_profiles
        self._init_neurons_from_profile()

    def add_neuron_group(self, neuron_profile: dict):
        """
        Add neuron group to existing configuration.

        Args:
            neuron_profile (dict): Additional neuron group configuration parameters.
        """
        assert neuron_profile is not None, "neuron_profile is None"
        self.neuron_profiles.update(neuron_profile)
        self.neurons.add_neuron_group(neuron_profile)

    def _init_neurons_from_profile(self):
        """
        Initialize the neurons class from the profile. 
        It must be called after the arena is set.
        """
        if self.neuron_profiles is not None and self.arena is not None:
            self.neurons.init_from_profile(self.neuron_profiles)

    # ================================
    # Behavior
    # ================================
    def random_traverse(self, duration: float, batch_size: int, 
                        init_pos=None, init_state=None, pause_prob=0, **kwargs):
        """
        Generate a random trajectory for the agent.

        Creates a trajectory using autonomous behavior with optional pausing.
        Updates the agent's internal state with the final state of the trajectory.
        
        Args:
            duration (float): Duration of the trajectory in seconds.
            batch_size (int): Number of parallel trajectories to generate.
            init_pos (np.ndarray, optional): Initial position of the agent.
            init_state (AgentState, optional): Initial state of the agent.
            pause_prob (float, optional): Probability of pausing the agent (default: 0).
                
        Returns:
            Trajectory: Generated trajectory data.
        """
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
        """
        Take a single controllable step. This is for controllable behavior.
        
        Updates the agent's state by applying the given displacement through
        the controllable behavior system.
        
        Args:
            displacement (np.ndarray): Displacement vector to apply to the agent.
        """
        self._state = self.behavior.controllable.step(self._state, displacement)

    def get_response(
            self,
            agent_data: Union[AgentState, Trajectory],
            return_format: str = 'array', 
            keys: list = None, 
            str_filter: str = None, 
            type_filter: str = None
        ):
        """
        Get neuronal responses for given agent data.

        Computes neuronal responses (e.g., place cell firing, grid cell activity)
        for the provided agent state or trajectory data.

        Args:
            agent_data (Union[AgentState, Trajectory]): Agent state or trajectory
                data to compute neuronal responses for.
            return_format (str, optional): Format for returned data (default: 'array').
            keys (list, optional): Specific neuronal keys to return.
            str_filter (str, optional): String filter for neuronal selection.
            type_filter (type, optional): Type filter for neuronal selection. The
                type filter check the neuron_type attribute of the corresponding 
                neuronal group.

        Returns:
            Neuronal response data in the specified format.

        Examples:
            >>> # Get only place cell responses
            >>> neuron_groups = {
            >>>     'place_cells': {
            >>>         'type': 'place_cells',
            >>>         'n_cells': 100,
            >>>     },
            >>>     'grid_cells': {
            >>>         'type': 'grid_cells',
            >>>         'n_cells': 100,
            >>>     },
            >>>     'weak_spatially_modulated': {
            >>>         'type': 'weak_sm_cell',
            >>>         'n_cells': 100,
            >>>     },
            >>> }
            >>> traj = agent.random_traverse(duration=10, batch_size=10)
            >>> responses = agent.get_response(traj, return_format='array', keys=['place_cells'])
            >>>
            >>> # Get all neuronal responses for cell groups with key words 'cell'
            >>> responses = agent.get_response(traj, return_format='array', str_filter='cell')
            >>>
            >>> # Get all neuronal responses for cell groups of type 'weak_sm_cell'
            >>> responses = agent.get_response(traj, return_format='array', type_filter='weak_sm_cell')
        """
        if isinstance(agent_data, RawAgentState):
            agent_data = agent_data.to_agent_state()
        elif isinstance(agent_data, RawTrajectory):
            agent_data = agent_data.to_trajectory()
        return self.neurons.get_response(agent_data, return_format, keys, str_filter, type_filter)

    def spawn(self, init_pos=None, init_state=None):
        """
        Spawn the agent at the given position and state.

        Resets the behavior and initializes the agent at a specific
        location and state for manual control.
        
        Args:
            init_pos (np.ndarray, optional): Initial position of the agent.
            init_state (AgentState or RawAgentState, optional): Initial state of the agent. 
                If both init_pos and init_state are provided, init_state will be used.
        """
        self.behavior.controllable.reset()
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
        Spawn the agent at a random position in the arena.
        
        Resets the controllable behavior and places the agent at a randomly
        selected valid position within the arena.
        
        Args:
            batch_size (int): Number of random positions to generate.
        """
        self.spawn(init_pos=self.arena.random_position(batch_size))
