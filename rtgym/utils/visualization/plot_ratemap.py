import numpy as np
import matplotlib.pyplot as plt


def plot_ratemaps(ratemaps, n_rows, n_cols, cmap="jet", per_image_height=2, arena_map=None):
    """
    Plots a grid of average rate maps with specified rows and columns,
    dynamically adjusting the figure size to maintain aspect ratios.
    
    Args:
        ratemaps: numpy array of shape (n, w, h)
        n_rows: number of rows in the subplot grid
        n_cols: number of columns in the subplot grid
        cmap: colormap to use for the images
        per_image_height: height of each individual subplot in inches (default: 2)
    """
    # If ratemaps is a Tensor, convert it to a numpy array
    if hasattr(ratemaps, 'cpu'):
        ratemaps = ratemaps.cpu().numpy()
    elif hasattr(ratemaps, 'numpy'):
        ratemaps = ratemaps.numpy()

    n_maps, h, w = ratemaps.shape  # This is a transposed shape
    total_plots = n_rows * n_cols

    if total_plots < n_maps:
        print(f"Warning: The grid ({n_rows}x{n_cols}) is smaller than the number of rate maps ({n_maps}). Some maps won't be displayed.")

    # Calculate aspect ratio of individual images
    image_aspect = w / h  # width divided by height

    # Calculate figure size
    # per_image_height is in inches
    per_image_width = per_image_height * image_aspect
    total_width = n_cols * per_image_width
    total_height = n_rows * per_image_height

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height))
    axes = axes.flatten()  # Flatten in case of multiple rows and columns

    # Plot each rate map
    for i in range(total_plots):
        ax = axes[i]
        if i < n_maps:
            ratemap = ratemaps[i]
            if arena_map is not None:
                # The arena_map corresponds to the inv_arena_map in gym.arena,
                # where 1 is free, 0 is wall.
                # If arena_map is not None, set wall to nan and use black as bad color
                ratemap = np.where(arena_map == 0, np.nan, ratemap)
            cm = plt.cm.get_cmap(cmap)
            cm.set_bad(color='grey')
            ax.imshow(ratemap, cmap=cm)
        else:
            ax.axis('off')  # Hide axes without data

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig, axes
