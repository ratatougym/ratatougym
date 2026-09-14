"""EMA navigation adapted from grasp-lyrl/grid_and_place's rtgym.

The motion equations and random draw order match its eager implementation.
The control device is configurable; no implicit compilation occurs.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from rtgym.dataclasses import AgentState, Trajectory


class TrajectoryGenerator:
    """Smoothed speed/direction targets with tangential boundary avoidance.

    ``spd_mean`` and ``spd_sd`` are in grid cells/second, as in grid_and_place.
    Use ``generate_trajectory`` for an explicit number of samples.
    """

    def __init__(self, gym, bhv_device=None):
        self.gym = gym
        device = gym.device if bhv_device is None else bhv_device
        self.device = torch.device(device)
        self.bhv_device = self.device
        self.cur_state = AgentState(device=self.device)
        self.initialized = False

    def init_from_profile(self, profile):
        self._set_config(profile)
        self._recompute_maps()
        self.reset()
        self.initialized = True

    def _set_config(self, profile):
        # Retain upstream defaults and the conversion used by boundary avoidance.
        self.spd_mean = float(profile['spd_mean'])
        spd_sd = profile.get('spd_sd', 0)
        self.spd_sd = float(spd_sd)
        alpha_spd = profile.get('alpha_spd', 0.1)
        self.alpha_spd = float(alpha_spd)
        alpha_dir = profile.get('alpha_dir', 0.1)
        self.alpha_dir = float(alpha_dir)
        self.switch_dir_prob = float(profile['switch_dir_prob'])
        self.switch_spd_prob = float(profile['switch_spd_prob'])
        look_around_scale = profile.get('look_around_scale', 0)
        self.look_around_scale = float(look_around_scale)
        if self.spd_mean <= 0 or self.spd_sd < 0:
            raise ValueError('spd_mean must be positive and spd_sd non-negative.')
        bounded = ('alpha_spd', 'alpha_dir', 'switch_dir_prob',
                   'switch_spd_prob', 'look_around_scale')
        for name in bounded:
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1].')

        self.avoid_boundary_dist = -1
        avoidance = profile.get('boundary_avoidance')
        if avoidance is not None:
            avoidance = float(avoidance)
            typical_speed = self.spd_mean * self.gym.t_res / 1e3
            distance = 12.0 * typical_speed * (0.5 + 0.5 * avoidance)
            self.avoid_boundary_dist = max(distance, 1e-6)

    def generate_trajectory(self, duration_ts, batch_size, init_state=None, init_pos=None):
        if not self.initialized:
            raise ValueError('Initialize the control profile before generating trajectories.')
        n_steps = duration_ts
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError('n_steps must be a positive integer.')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer.')
        ndim = self.gym.arena.ndim

        # Precompute targets before sampling starting positions, matching upstream.
        self._precompute_targets(batch_size, n_steps)
        state = self._init_state(batch_size, init_state, init_pos)
        coord = torch.zeros(batch_size, n_steps, ndim, device=self.device)
        spd = torch.zeros(batch_size, n_steps, 1, device=self.device)
        mv_dir = torch.zeros(batch_size, n_steps, ndim, device=self.device)
        head_dir = torch.zeros(batch_size, n_steps, ndim, device=self.device)

        # Record the initial state, then apply one motion update per timestep.
        for ts in range(n_steps):
            if ts > 0:
                self._update_targets(state, ts)
                self._update_dynamics(state)
                self._avoid_boundary(state)
                proposed = state.coord + state.disp
                indices = proposed.long()
                valid = self.gym.arena.validate_index(indices)
                proposed[~valid] = state.coord[~valid]
                state.coord = proposed
                state.head_dir = self.head_dirs[:, ts]
            coord[:, ts] = state.coord
            spd[:, ts] = state.spd
            mv_dir[:, ts] = state.mv_dir
            head_dir[:, ts] = state.head_dir

        self.cur_state = state
        traj = Trajectory(coord=coord, spd=spd, mv_dir=mv_dir, head_dir=head_dir, device=self.device)
        return traj, state

    def _precompute_targets(self, batch_size, n_steps):
        # Draw speed targets and switching masks in the original order.
        shape = (batch_size, n_steps)
        spd_draws = torch.rand(shape, device=self.device)
        self.switch_spd_mask = spd_draws < self.switch_spd_prob
        speed_shape = (batch_size, n_steps, 1)
        speeds = self._random_speeds(speed_shape)
        self.target_spds = speeds / 1e3 * self.gym.t_res

        # Normalize random direction targets before drawing head variations.
        dir_draws = torch.rand(shape, device=self.device)
        self.switch_dir_mask = dir_draws < self.switch_dir_prob
        direction_shape = (batch_size, n_steps, self.gym.arena.ndim)
        dirs = torch.randn(direction_shape, device=self.device)
        norms = dirs.norm(dim=-1, keepdim=True)
        norms = norms.clamp_min(1e-8)
        self.target_dirs = dirs / norms
        self.head_dirs = self._random_head_dirs(self.target_dirs)

    def _random_speeds(self, shape):
        if self.spd_sd == 0:
            return torch.full(shape, self.spd_mean, device=self.device)

        # LogNormal samples on CPU upstream, even when targets live on CUDA.
        mean, sd = self.spd_mean, self.spd_sd
        variance_ratio = 1 + sd**2 / mean**2
        variance_ratio = torch.tensor(variance_ratio, device=self.device)
        log_variance = torch.log(variance_ratio)
        sigma = log_variance ** 0.5
        sigma = sigma.item()
        mean_ratio = mean**2 / (mean**2 + sd**2) ** 0.5
        mean_ratio = torch.tensor(mean_ratio, device=self.device)
        mu = torch.log(mean_ratio)
        mu = mu.item()
        distribution = torch.distributions.LogNormal(mu, sigma)
        samples = distribution.sample(shape)
        return samples.to(self.device)

    def _random_head_dirs(self, dirs):
        batch_size, n_steps, ndim = dirs.shape
        scale = self.look_around_scale
        if ndim == 2:
            if scale > 0:
                variation = torch.randn(batch_size, n_steps, 1, device=self.device)
                variation = variation * scale
                variation = variation.transpose(1, 2)
                kernel = torch.ones(1, 1, 5, device=self.device)
                kernel = kernel / 5
                variation = F.conv1d(variation, kernel, padding=2)
                variation = variation.transpose(1, 2)
            else:
                variation = torch.zeros(batch_size, n_steps, 1, device=self.device)
            angle = torch.atan2(dirs[:, :, 1], dirs[:, :, 0])
            variation = variation.squeeze(-1)
            angle = angle + variation
            cosine = torch.cos(angle)
            sine = torch.sin(angle)
            return torch.stack([cosine, sine], dim=-1)

        # In 3D, smooth each noise component along time, then normalize vectors.
        if scale == 0:
            return dirs.clone()
        noise = torch.randn(batch_size, n_steps, ndim, device=self.device)
        noise = noise * scale
        kernel = torch.ones(1, 1, 5, device=self.device)
        kernel = kernel / 5
        for axis in range(ndim):
            component = noise[:, :, axis:axis + 1]
            component = component.transpose(1, 2)
            component = F.conv1d(component, kernel, padding=2)
            noise[:, :, axis:axis + 1] = component.transpose(1, 2)
        head_dir = dirs + noise
        norms = head_dir.norm(dim=-1, keepdim=True)
        norms = norms.clamp_min(1e-8)
        return head_dir / norms

    def _init_state(self, batch_size, init_state, init_pos):
        # Continue a full EMA state; positions alone create fresh motion targets.
        if init_state is None:
            state = AgentState(device=self.device)
            if init_pos is not None:
                state.coord = torch.as_tensor(init_pos, device=self.device)
        else:
            if not isinstance(init_state, AgentState):
                raise TypeError('EMA continuation requires a AgentState.')
            state = init_state.clone()
            state.to(self.device)
        if state.coord is None:
            free_space = self.gym.arena.free_space_numpy
            n_free = len(free_space)
            indices = torch.randint(n_free, (batch_size,), device=self.device)
            free_space = torch.as_tensor(free_space, device=self.device)
            state.coord = free_space[indices].float()
        state.coord = state.coord.float()
        expected_shape = (batch_size, self.gym.arena.ndim)
        if state.coord.shape != expected_shape:
            raise ValueError(f'Initial coordinates must have shape {expected_shape}.')
        indices = state.coord.long()
        valid = self.gym.arena.validate_index(indices)
        if not valid.all():
            raise ValueError('Initial coordinates must be in free space.')
        if state.spd is None or state.spd_target is None:
            state.spd = self.target_spds[:, 0]
            state.spd_target = self.target_spds[:, 0]
        if state.mv_dir is None or state.mv_dir_target is None:
            state.mv_dir = self.target_dirs[:, 0]
            state.mv_dir_target = self.target_dirs[:, 0]
        if state.head_dir is None:
            state.head_dir = self.head_dirs[:, 0]
        return state

    def _update_targets(self, state, ts):
        spd_mask = self.switch_spd_mask[:, ts]
        if spd_mask.any():
            state.spd_target[spd_mask] = self.target_spds[spd_mask, ts]
        dir_mask = self.switch_dir_mask[:, ts]
        if dir_mask.any():
            state.mv_dir_target[dir_mask] = self.target_dirs[dir_mask, ts]

    def _update_dynamics(self, state):
        state.spd = (1.0 - self.alpha_spd) * state.spd + self.alpha_spd * state.spd_target
        direction = (1.0 - self.alpha_dir) * state.mv_dir + self.alpha_dir * state.mv_dir_target
        norms = direction.norm(dim=-1, keepdim=True)
        norms = norms.clamp_min(1e-8)
        state.mv_dir = direction / norms

    def _avoid_boundary(self, state):
        if self.avoid_boundary_dist <= 0:
            return
        indices = []
        for axis, size in enumerate(self.gym.arena.dimensions):
            index = state.coord[:, axis].long()
            index = index.clamp(0, size - 1)
            indices.append(index)
        indices = tuple(indices)
        coefficient = self.distance_map[indices]
        active = coefficient > 1e-2
        if not active.any():
            return

        # Project active directions onto the local wall tangent plane.
        direction = state.mv_dir[active]
        active_indices = tuple(index[active] for index in indices)
        normal = self.normal_map[active_indices]
        product = direction * normal
        dot = product.sum(dim=-1)
        toward_wall = dot < 0
        toward_wall = toward_wall.float()
        dot_column = dot.unsqueeze(-1)
        tangent = direction - dot_column * normal
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        tangent_norm = tangent_norm.clamp_min(1e-8)
        tangent = tangent / tangent_norm

        # Blend more strongly near walls and at higher speeds.
        spd = state.spd[active]
        spd = spd.squeeze(-1)
        mean_spd = self.spd_mean / 1e3 * self.gym.t_res
        speed_scale = spd / (mean_spd + 1e-8)
        speed_scale = speed_scale.clamp(0.5, 3.0)
        blend = coefficient[active] * toward_wall * speed_scale
        blend = blend.unsqueeze(-1)
        direction = (1 - blend) * direction + blend * tangent
        norms = direction.norm(dim=-1, keepdim=True)
        norms = norms.clamp_min(1e-8)
        state.mv_dir[active] = direction / norms

    def _recompute_maps(self):
        if self.avoid_boundary_dist <= 0:
            return
        free_mask = self.gym.arena.inv_arena_map.astype(np.uint8)
        distance = distance_transform_edt(free_mask)
        distance = distance.astype(np.float32)
        exponent = -(distance**2 / self.avoid_boundary_dist)
        distance_map = np.exp(exponent)
        gradients = np.gradient(distance)
        normal_map = np.stack(gradients, axis=-1)
        norms = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normal_map = normal_map / norms
        self.distance_map = torch.as_tensor(distance_map, device=self.device)
        self.normal_map = torch.as_tensor(normal_map, device=self.device)

    def reset(self):
        self.cur_state = AgentState(device=self.device)

    def step(self, state, mv_dir, spd, head_dir):
        """Apply commanded grid-cell displacement with the same wall rejection."""
        if state is None or state.coord is None:
            raise ValueError('Spawn an agent before stepping.')
        state = state.clone()
        state.to(self.device)
        state.mv_dir = torch.as_tensor(mv_dir, device=self.device)
        state.spd = torch.as_tensor(spd, device=self.device)
        state.head_dir = torch.as_tensor(head_dir, device=self.device)
        expected = state.coord.shape
        if state.mv_dir.shape != expected or state.head_dir.shape != expected:
            raise ValueError('Direction vectors must match the coordinate shape.')
        if state.spd.shape != (expected[0], 1):
            raise ValueError('spd must have shape (batch, 1).')
        proposed = state.coord + state.disp
        indices = proposed.long()
        valid = self.gym.arena.validate_index(indices)
        proposed[~valid] = state.coord[~valid]
        state.coord = proposed
        return state
