"""Agent interface following grid_and_place's control and neuron API."""

import torch
from rtgym.dataclasses import AgentState
from rtgym.agent.behavior import Behavior
from rtgym.agent.control import TrajectoryGenerator
from rtgym.agent.neurons import Neurons


class Agent:
    def __init__(self, gym, device=None):
        self.gym = gym
        self.device = gym.device if device is None else torch.device(device)
        self.control_profile = None
        self.neuron_profiles = None
        self.neurons = Neurons(gym)
        self.control = None

    @property
    def arena(self):
        return self.gym.arena

    @property
    def neuron_groups(self):
        return self.neurons.neuron_groups

    @property
    def state(self):
        if self.control is None:
            raise ValueError('Control is not initialized; call init_control first.')
        return self.control.cur_state

    def num_neurons(self, keys=None, str_filter=None, type_filter=None):
        return self.neurons.num_neurons(keys, str_filter, type_filter)

    def init_control(self, control_profile):
        """Initialize trajectory_generator (EMA) or the original random_walk."""
        profile = dict(control_profile)
        control_type = profile.get('control_type', 'trajectory_generator')
        if control_type == 'trajectory_generator':
            device = profile.get('device', self.device)
            control = TrajectoryGenerator(self.gym, bhv_device=device)
            control.init_from_profile(profile)

        # Use the retained NumPy model when explicitly requested.
        elif control_type == 'random_walk':
            control = Behavior(self.gym)
            control.init_from_profile(profile)
        else:
            raise ValueError(f'Unknown control_type: {control_type}')

        # Publish the initialized control and its profile.
        self.control = control
        self.control_profile = profile

    def init_neurons(self, neuron_profiles):
        """Initialize named neuron groups; missing type selects diffusion_cell."""
        self.neurons.init_from_profile(neuron_profiles)
        self.neuron_profiles = {}
        for key, profile in neuron_profiles.items():
            self.neuron_profiles[key] = dict(profile)

    def add_neuron_group(self, neuron_profile):
        self.neurons.add_neuron_group(neuron_profile)
        if self.neuron_profiles is None:
            self.neuron_profiles = {}
        for key, profile in neuron_profile.items():
            self.neuron_profiles[key] = dict(profile)

    def _on_arena_change(self):
        # Rebuild derived maps and discard state tied to the previous arena.
        if self.control_profile is not None:
            self.init_control(self.control_profile)
        if self.neuron_profiles is not None:
            self.init_neurons(self.neuron_profiles)

    def random_traverse(self, duration_ts, batch_size, init_state=None, pause_prob=0, **kwargs):
        """Generate a tensor trajectory of duration_ts samples, continuing state."""
        if not isinstance(duration_ts, int) or duration_ts < 1:
            raise ValueError('duration_ts must be a positive integer number of timesteps.')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')

        # Validate optional behavior before advancing state.
        if not 0 <= pause_prob <= 1:
            raise ValueError('pause_prob must be in [0, 1].')
        if kwargs:
            names = tuple(kwargs)
            raise TypeError(f'Unknown trajectory options: {names}')

        # Continue from the last state unless another was supplied.
        state = self.state
        if init_state is None and state is not None and state.coord is not None:
            init_state = state
        traj, state = self.control.generate_trajectory(duration_ts, batch_size, init_state)

        # Pause complete trials and align their stored continuation coordinates.
        if pause_prob > 0:
            draws = torch.rand(batch_size, device=traj.device)
            paused = draws < pause_prob
            start_pos = traj.coord[paused, 0]
            traj.coord[paused] = start_pos[:, None]

            # Keep motion and continuation state aligned with paused trials.
            traj.spd[paused] = 0
            traj.head_dir[paused] = 0
            state.coord[paused] = start_pos
            state.spd[paused] = 0
            if isinstance(self.control, Behavior):
                self.control.raw_state = None

        # Publish the final state and move the trajectory to the output device.
        self.control.cur_state = state
        return traj.to(self.device)

    def get_response(self, agent_data, return_format='tensor', keys=None,
                     str_filter=None, type_filter=None, device=None):
        return self.neurons.get_response(agent_data, return_format, keys, str_filter, type_filter, device)

    def spawn(self, init_state=None):
        if self.control is None:
            raise ValueError('Control is not initialized; call init_control first.')
        if not isinstance(init_state, AgentState):
            raise ValueError('Provide an AgentState or use random_spawn.')
        self.control.reset()
        self.control.cur_state = init_state.clone()
        return self.control.cur_state

    def random_spawn(self, batch_size):
        coord = self.arena.generate_random_pos(batch_size)
        coord = coord.float()
        state = AgentState(coord=coord, device=self.device)
        return self.spawn(state)

    def step(self, mv_dir, spd, head_dir):
        state = self.step_state(self.state, mv_dir, spd, head_dir)
        self.control.cur_state = state
        return state

    def step_state(self, state, mv_dir, spd, head_dir):
        if self.control is None:
            raise ValueError('Control is not initialized; call init_control first.')
        return self.control.step(state, mv_dir, spd, head_dir)
