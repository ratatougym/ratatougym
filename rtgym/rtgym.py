"""
RatatouGym is a python package that provides a simple interface to generate sensory responses
from a virtual agent in a virtual environment.

This file contains more documentation as it serves as the main API for the package.
"""

import os
import numpy as np
from rtgym.agent import Agent
from rtgym.arena import Arena
from rtgym.dataclasses import Trajectory
import torch
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation


class RatatouGym():
    """
    The RatatouGym class is a singleton that integrates the environment, agent, behavior, and sensory streams.
    When initialized, the class creates an arena and an uninitialized agent.
    The arena is the environment the agent moves in, and the agent is the subject navigating the arena.

    After initialization, both the arena and the agent must be set up.
    For the arena, you can either choose from predefined shapes or provide a custom 2D NumPy array as the map.
    For the agent, you must configure sensory and behavioral parameters. Sensory is defined by a dictionary of the sensory cell types you want to simulate. Behavior is defined by parameters such as movement speed, how frequently the agent changes direction, and how it avoids obstacles.
    
    Args:
        temporal_resolution (float): Time resolution in milliseconds. E.g. 50 will mean 50ms per timestep.
        spatial_resolution (float): Spatial resolution in units per pixel. E.g. 1 will mean 1cm per pixel.
        **kwargs: Additional keyword arguments.

    Example:
        >>> from rtgym import RatatouGym
        >>> gym = RatatouGym(
        ...     temporal_resolution=50,
        ...     spatial_resolution=1,
        ... )
    """
    def __init__(
            self,
            temporal_resolution,
            spatial_resolution,
            device='cpu',
            **kwargs
        ):
        self.device = torch.device(device)
        self.temporal_resolution = temporal_resolution
        self.spatial_resolution = spatial_resolution
        self.arena = Arena(self)
        self.agent = Agent(self)

        self.arena.subscribe(self._on_arena_change)

    # =========================================================================
    # Basic get methods without trial generation
    # =========================================================================
    @property
    def t_res(self):
        """
        Get temporal resolution of the gym. The temporal resolution controls how
        continuous time in the real world is discretized into timesteps.
        
        Returns:
            float: Temporal resolution in milliseconds.
        """
        return self.temporal_resolution

    @property
    def s_res(self):
        """
        Get spatial resolution of the gym. The spatial resolution controls how
        the arena is discretized into pixels. The units are in cm per pixel.
        
        Returns:
            float: Spatial resolution in cm per pixel.
        """
        return self.spatial_resolution

    def to_ts(self, t):
        """
        Convert time (in seconds) to number of timesteps. This is a helper function.
        Note that this method will round the time to the nearest integer.

        Args:
            t (int, float, np.ndarray, or torch.Tensor): Time value(s) in seconds
            
        Returns:
            int, np.ndarray, or torch.Tensor: Corresponding number of timestep(s)

        Raises:
            TypeError: If input type is not supported.

        Example:
            >>> gym = RatatouGym(temporal_resolution=50, spatial_resolution=1)
            >>> gym.to_ts(1)
            >>> 20  # 1 second = 20 timesteps
            >>> gym.to_ts(np.array([1, 2, 3]))
            >>> array([20 40 60])
            >>> gym.to_ts(torch.tensor([1, 2, 3]))
            >>> tensor([20, 40, 60], dtype=torch.int32)
        """
        if isinstance(t, (int, float)):
            return int(np.round(t * 1e3 / self.t_res))
        elif isinstance(t, np.ndarray):
            return np.round(t * 1e3 / self.t_res).astype(int)
        elif 'torch' in str(type(t)):
            return torch.round(t * 1e3 / self.t_res).int()
        else:
            raise TypeError("Unsupported type for t")

    def to_sec(self, ts):
        """
        Convert timesteps to seconds. This is a helper function to convert timesteps in milliseconds
        to seconds.
        
        Args:
            ts (int, float, np.ndarray, or torch.Tensor): Timestep value(s).
            
        Returns:
            int, float, np.ndarray, or torch.Tensor: Corresponding time(s) in seconds.

        Raises:
            TypeError: If input type is not supported.

        Example:
            >>> gym = RatatouGym(temporal_resolution=50, spatial_resolution=1)
            >>> gym.to_sec(20)
            >>> 1.0  # 20 timesteps = 1 second (50ms per timestep)
            >>> gym.to_sec(np.array([20, 40, 60]))
            >>> array([1., 2., 3.])
            >>> gym.to_sec(torch.tensor([20, 40, 60]))
            >>> tensor([1., 2., 3.])
        """
        if isinstance(ts, (int, float)):
            return ts * self.t_res / 1e3
        elif isinstance(ts, np.ndarray):
            return ts * self.t_res / 1e3
        elif 'torch' in str(type(ts)):
            return ts * self.t_res / 1e3
        else:
            raise TypeError("Unsupported type for ts")

    def to_coord(self, pos):
        """
        Convert position to coordinate indices. This method transforms continuous
        position values into discrete coordinate indices based on the spatial
        resolution of the arena.
        
        Args:
            pos (array-like): Position values in the continuous space of the arena.
            
        Returns:
            np.ndarray: Coordinate indices in the discretized arena grid.
        """
        return np.array(pos/self.s_res)

    @property
    def arena_map(self):
        """
        Get the arena map. The arena map represents the spatial layout of the
        environment, encoding walls and free space for navigation.
        
        Returns:
            np.ndarray: Arena map of shape (H, W) where 0 represents free space 
                and 1 represents walls. This binary representation is used 
                internally for collision detection and path planning.
        """
        return self.arena.arena_map
    
    @property
    def inv_arena_map(self):
        """
        Get the inverted arena map for visualization. The inverted map is useful
        for plotting and visualization purposes where free space should be
        highlighted rather than walls.
        
        Returns:
            np.ndarray: Inverted arena map of shape (H, W) where 1 represents 
                free space and 0 represents walls. This format is optimized 
                for visual display and matplotlib visualization.
        """
        return self.arena.inv_arena_map

    def random_pos(self, n):
        """
        Generate random valid positions in the arena. This method creates random
        positions that are guaranteed to be in free space (not inside walls),
        making them suitable for agent placement or target positioning.
        
        Args:
            n (int): Number of random positions to generate. Must be a positive integer.
            
        Returns:
            np.ndarray: Array of shape (n, 2) containing random positions.
        """
        return self.arena.generate_random_pos(n)
    # =========================================================================
    
    # =========================================================================
    # Set methods to initialize the arena and agent
    # =========================================================================
    def _on_arena_change(self):
        """
        Callback method triggered when arena changes. This internal method ensures
        proper synchronization between the arena and agent components when the
        arena configuration is modified.
        
        The method automatically notifies the agent component about arena changes,
        allowing it to update its internal state and sensory configurations
        accordingly. This ensures that the agent's spatial representations
        remain consistent with the current arena layout.
        
        Note:
            This is an internal callback method and should not be called directly
            by users. It is automatically invoked when arena properties change.
        """
        self.agent._on_arena_change()

    def load_arena(self, path):
        """
        Load an arena configuration from a file. This method allows loading
        previously saved arena configurations, enabling reproducible experiments
        and sharing of environment setups.
        
        The loaded arena will replace the current arena configuration entirely,
        including its spatial layout, dimensions, and any associated metadata.
        The agent will be automatically notified of the arena change through
        the internal callback system.
        
        Args:
            path (str): Path to the arena file. Should be a valid file path
                pointing to a previously saved arena configuration. The file
                format depends on the arena implementation but typically uses
                .npz format for numpy arrays.
                
        Raises:
            FileNotFoundError: If the specified path does not exist.
            ValueError: If the file format is incompatible or corrupted.
            
        Example:
            >>> gym = RatatouGym(temporal_resolution=50.0, spatial_resolution=1.0)
            >>> gym.load_arena('saved_environments/maze_1.npz')
        """
        self.arena.load(path)

    def save_arena(self, path):
        """
        Save the current arena configuration to a file. This method enables
        persistence of arena configurations for later use, sharing, or
        experimental reproducibility.
        
        The saved file will contain the complete arena state including the
        spatial layout, dimensions, and any associated metadata. The saved
        arena can be later loaded using the load_arena method.
        
        Args:
            path (str): Path where the arena file will be saved. Must have
                .npz extension as the arena is saved in numpy's compressed
                format for efficiency and compatibility.
                
        Raises:
            AssertionError: If the file extension is not .npz.
            PermissionError: If the specified path is not writable.
            OSError: If there are issues with file system operations.
            
        Example:
            >>> gym = RatatouGym(temporal_resolution=50.0, spatial_resolution=1.0)
            >>> gym.init_arena_map(shape='square', width=100, height=100)
            >>> gym.save_arena('my_environments/custom_maze.npz')
        """
        # file extension is npz
        assert os.path.splitext(path)[1] == '.npz', "file extension must be .npz"
        self.arena.save(path)

    def init_arena_map(self, **kwargs):
        """
        Initialize the arena map with a predefined geometric shape. This method
        calls the corresponding function in the arena class to initialize the navigation space from a set of predefined shapes.
        
        The method supports various predefined shapes commonly used in spatial
        navigation research, such as squares, rectangles, circles, and more
        complex geometries. Each shape type accepts specific parameters that
        define its dimensions and characteristics. Please refer to the `rtgym.arena.arena_shapes` module for more details.

        Args:
            **kwargs: Shape-specific parameters that vary depending on the
                selected arena geometry. Each arean shape might have different parameters. 
                Please refer to the `rtgym.arena.arena_shapes` module for more details.
                
        Example:
            >>> gym = RatatouGym(temporal_resolution=50.0, spatial_resolution=1.0)
            >>> gym.init_arena_map(shape='rectangle', dimensions=[100, 100])
        """
        self.arena.init_arena_map(**kwargs)

    def set_arena_map(self, arena_map):
        """
        Set arena map with a custom binary layout. This method allows complete
        control over the arena geometry by directly specifying the spatial
        layout as a binary array.
        
        The custom arena map enables creation of arbitrary environments beyond
        the predefined shapes, allowing for complex mazes, irregular boundaries,
        or specialized experimental setups. The binary format uses 0 for free
        space and 1 for walls, following standard conventions in spatial
        navigation research.
        
        Args:
            arena_map (np.ndarray): Custom arena map of shape (H, W) where
                0 represents free space (navigable areas) and 1 represents
                walls (impassable barriers). The array should be 2-dimensional
                with consistent data types (typically bool or int).
                
        Raises:
            ValueError: If arena_map is not a 2D array or contains invalid values.
            TypeError: If arena_map is not a numpy array or compatible type.
            
        Example:
            >>> import numpy as np
            >>> custom_map = np.zeros((50, 50))  # 50x50 open field
            >>> custom_map[0, :] = 1  # top wall
            >>> custom_map[-1, :] = 1  # bottom wall
            >>> custom_map[:, 0] = 1  # left wall
            >>> custom_map[:, -1] = 1  # right wall
            >>> gym.set_arena_map(custom_map)
        """
        self.arena.set_arena_map(arena_map=arena_map)
    # ======================================================================================
    
    # ======================================================================================
    # Sensory methods
    # ======================================================================================
    def set_sensory_manually(self, sens_type, sensory):
        import warnings
        warnings.warn("DeprecationWarning: The method 'set_sensory_manually' is deprecated. "
                      "Potential implementation issues might cause unintended behavior. "
                      "It is recommended to implement a separate sensory class instead.")
        # self.agent.set_sensory_manually(sens_type, sensory)
    # ======================================================================================

    # ======================================================================================
    # Visualization methods
    # ======================================================================================
    @staticmethod
    def _compute_plot_dimensions(arena_map, fig_height):
        aspect_ratio = arena_map.shape[1] / arena_map.shape[0]
        fig_width = fig_height * aspect_ratio
        return fig_width, fig_height

    def vis_gif(self, traj, plot_w, plot_h, return_format='anim', stride=10, interval=50, max_frames=100):
        """Create an animated visualization of trajectory and heading direction.
        
        This method generates an animated plot showing the agent's path through the arena
        with real-time visualization of position and heading direction. The trajectory
        is displayed as a red line that progressively reveals the path taken, while a
        blue arrow indicates the agent's current heading direction at each timestep.
        
        The animation can be customized for performance by adjusting frame rate, sampling
        density, and maximum duration. For long trajectories, the stride and max_frames
        parameters help balance visualization quality with computational efficiency.
        
        Args:
            traj: Trajectory data object containing position coordinates (int_coord, float_coord)
                and heading direction (hd) information. Must have shape (batch, timesteps, dims).
            plot_w (float): Desired width of the plot in inches. Will be automatically 
                computed based on arena aspect ratio if not specified correctly.
            plot_h (float): Desired height of the plot in inches.
            return_format (str): Output format for the animation. Options:
                - 'anim': Returns matplotlib animation object for further manipulation
                - 'html': Returns HTML representation for Jupyter notebook display
            stride (int): Step size for sampling trajectory points. Higher values create
                faster animations by skipping frames but may lose trajectory detail.
                Default is 10.
            interval (int): Time between animation frames in milliseconds. Lower values
                create faster playback. Default is 50ms (20 FPS).
            max_frames (int): Maximum number of frames to generate regardless of trajectory
                length. Helps prevent memory issues with very long trajectories and
                ensures reasonable animation generation times. Default is 100.
                
        Returns:
            matplotlib.animation.FuncAnimation: Animation object that can be saved,
                displayed, or converted to different formats.
                
        Raises:
            ValueError: If trajectory data is empty or malformed (no data along expected axis).
            
        Example:
            >>> # Create animation for a recorded trajectory
            >>> anim = gym.vis_gif(trajectory, plot_w=8, plot_h=6, 
            ...                    stride=5, interval=100, max_frames=50)
            >>> anim.save('trajectory.gif', writer='pillow', fps=10)
        """
        if isinstance(traj, Trajectory):
            traj = traj.as_numpy()
        if traj.int_coord.shape[1] == 0 or traj.hd.shape[1] == 0:
            raise ValueError("traj.int_coord or traj.hd has no data along the expected axis.")

        plot_w, plot_h = self._compute_plot_dimensions(self.arena.arena_map, plot_h)
        fig, ax = plt.subplots(figsize=(plot_w, plot_h))
        ax.imshow(self.inv_arena_map)
        ax.axis('off')
        line, = ax.plot([], [], 'r')
        
        # Initialize the quiver arrow
        arrow = ax.quiver(0, 0, 0.05, 0.05, angles='xy', scale_units='xy', scale=0.2, color='blue')

        # Precompute trajectory data
        traj_length = traj.int_coord.shape[1]
        
        # Calculate appropriate stride to ensure we cover the full trajectory
        total_frames = min(max_frames, int(np.ceil(traj_length / stride)))
        frames_stride = max(1, int(np.ceil(traj_length / total_frames)))
        
        # Precompute frame indices to ensure we show the full trajectory
        frame_indices = np.linspace(0, traj_length-1, total_frames, dtype=int)
        
        def init():
            line.set_data([], [])
            arrow.set_offsets([0, 0])
            arrow.set_UVC(0.05, 0.05)
            return line, arrow

        def animate(i):
            frame_idx = frame_indices[i]
            
            # Always show the complete trajectory up to the current point
            line.set_data(traj.float_coord[0, :frame_idx+1, 1], traj.float_coord[0, :frame_idx+1, 0])

            if frame_idx > 0:
                # Get the tip position of the line
                tip_x, tip_y = traj.float_coord[0, frame_idx, 1], traj.float_coord[0, frame_idx, 0]
                
                # Calculate direction based on the angle
                angle = traj.hd[0, frame_idx, 0]
                direction_x = np.cos(angle)
                direction_y = np.sin(angle)
                
                # Update arrow position and direction
                arrow.set_offsets([tip_x, tip_y])
                arrow.set_UVC(direction_x, direction_y)
            
            return line, arrow

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=total_frames, interval=interval, blit=True)
        plt.tight_layout()

        return HTML(anim.to_jshtml()) if return_format == 'html' else anim

    def vis_traj(self, traj, cmap="viridis", linewidth=1, return_format=None, height=3, ax=None, vis_kwargs=None):
        """
        Visualize the trajectory of the agent in the arena.
        
        This method provides flexible visualization of agent trajectories, supporting both
        static plotting and animated displays. The trajectory is overlaid on the arena map,
        showing the path taken by the agent through the environment. The visualization
        automatically scales to maintain proper aspect ratios and provides options for
        customization through various parameters.
        
        For static visualizations, the complete trajectory is displayed as a line plot
        over the arena. For animated visualizations, the trajectory is built up frame
        by frame, optionally showing the agent's heading direction as it moves.

        Args:
            traj (rtgym.dataclass.Trajectory): Trajectory object containing the agent's
                position data and other trajectory information to be visualized.
            cmap (str, optional): Colormap for the arena background. Defaults to "viridis".
            linewidth (float, optional): Width of the trajectory line. Defaults to 1.
            return_format (str, optional): Format of the returned visualization. Options are:
                - 'anim': Returns matplotlib animation object
                - 'html': Returns HTML representation of animation for Jupyter notebooks
                - None: Returns static plot as (fig, ax) tuple
            height (float, optional): Height of the figure in inches. Width is automatically
                calculated to maintain arena aspect ratio. Defaults to 3.
            ax (matplotlib.axes.Axes, optional): Existing axes to plot on. If provided,
                figure size parameters are ignored.
            vis_kwargs (dict, optional): Additional keyword arguments passed to animation
                function when return_format is 'anim' or 'html'.
                
        Returns:
            tuple or animation object: 
                - If return_format is None: Returns (fig, ax) tuple for static plot
                - If return_format is 'anim': Returns matplotlib FuncAnimation object
                - If return_format is 'html': Returns HTML object for Jupyter display
        """

        if isinstance(traj, Trajectory):
            traj = traj.as_numpy()
        plot_w, plot_h = self._compute_plot_dimensions(self.arena.arena_map, height)
        if return_format == 'anim' or return_format == 'html':
            return self.vis_gif(traj, plot_w, plot_h, return_format, **vis_kwargs)
        else:
            fig, ax = plt.subplots(figsize=(plot_w, plot_h))
            ax.imshow(self.arena.inv_arena_map, cmap=cmap)
            ax.plot(traj.float_coord[:, :, 1].T, traj.float_coord[:, :, 0].T, color='red', linewidth=linewidth)
            ax.set_axis_off()
            return fig, ax
