import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np


def prepare_fields(fields, mask=None, bad_color='black', cmap='jet'):
    """
    Prepares the fields by applying a mask and setting up the colormap.
    """
    if not isinstance(fields, list):
        fields = [fields]
    
    cmap = plt.get_cmap(cmap)
    cmap.set_bad(color=bad_color)
    
    if mask is not None:
        fields = [np.where(mask, field, np.nan) for field in fields]
    
    return fields, cmap


def compute_layout(fields, n_cols):
    """
    Computes the layout for the grid of plots.
    """
    n_cells = fields[0].shape[0]
    aspect_ratio = fields[0].shape[1] / fields[0].shape[2]
    n_rows = -(-n_cells // n_cols)  # Ceiling division
    fig_width = n_cols * 2
    fig_height = n_rows * 2 * aspect_ratio
    return n_cells, n_rows, fig_width, fig_height


def plot_single_field(ax_, field, cmap, plot_option, min_max, title):
    """
    Plots a single firing field on the given axis.
    """
    if plot_option == 'rescale':
        field = (field - np.nanmin(field)) / (np.nanmax(field) - np.nanmin(field))
    elif plot_option == 'zero_center':
        max_abs = max(np.nanmax(field), abs(np.nanmin(field)))
        min_max = [-max_abs, max_abs]
    
    ax_.imshow(field, cmap=cmap, interpolation='none', 
               vmin=min_max[0] if min_max else None, 
               vmax=min_max[1] if min_max else None)
    ax_.set_title(title, fontsize=6)
    ax_.axis('off')


def plot_dual_fields(ax_, field_0, field_1, title):
    """
    Plots dual firing fields on the given axis.
    """
    ax_.imshow(field_0, extent=[0, 100, 0, 100], origin='upper', alpha=0.5, cmap='Reds')
    ax_.imshow(field_1, extent=[0, 100, 0, 100], origin='upper', alpha=0.5, cmap='Blues')
    ax_.set_title(title, fontsize=6)
    ax_.axis('off')


def generate_titles(fields, titles):
    """
    Generates titles for the plots.
    """
    if titles is None:
        return [f'Cell {i} fr (Hz)' for i in range(fields[0].shape[0])]
    return titles


def visualize_fields(
    fields, 
    titles=None, 
    n_cols=10, 
    cmap='jet', 
    mask=None, 
    bad_color='black', 
    plot_func=None, 
    title_func=None
):
    """
    Visualizes fields of cells using a custom plotting and title generation function.

    Args:
        fields (list or array): Fields to be visualized.
        titles (list, optional): Titles for each field. Default is None.
        n_cols (int): Number of columns in the grid layout. Default is 5.
        cmap (str): Colormap for visualization. Default is 'jet'.
        mask (array, optional): Mask to apply to fields, setting masked areas to NaN. Default is None.
        bad_color (str): Color for masked regions. Default is 'black'.
        plot_func (callable): Function to handle the plotting for each field.
        title_func (callable): Function to generate titles for each field.
    
    Returns:
        fig, ax: Matplotlib figure and axes objects.
    """
    # Prepare fields and colormap
    fields, cmap = prepare_fields(fields, mask, bad_color, cmap)
    
    # Compute layout dimensions
    n_cells, n_rows, fig_width, fig_height = compute_layout(fields, n_cols)
    
    # Create figure and axes
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), dpi=200)
    ax = ax.flatten() if n_rows > 1 else np.array(ax)

    # Default to single field plot if no plot_func is provided
    if plot_func is None:
        plot_func = plot_single_field
    if title_func is None:
        title_func = title_single_field

    # Plot each field
    for i in range(n_cells):
        field = fields[0][i] if isinstance(fields[0], list) else fields[0][i]
        title = title_func(i, field) if titles is None else titles[i]
        plot_func(ax[i], field, cmap, title)

    # Turn off unused axes
    for i in range(n_cells, len(ax)):
        ax[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_single_field(ax_, field, cmap, title):
    """
    Default function for plotting a single field.
    """
    ax_.imshow(field, cmap=cmap, interpolation='none')
    ax_.set_title(title, fontsize=6)
    ax_.axis('off')


def plot_dual_field(ax_, field, cmap, title):
    """
    Example dual field plotting function.
    """
    ax_.imshow(field[0], extent=[0, 100, 0, 100], origin='upper', alpha=0.5, cmap='Reds')
    ax_.imshow(field[1], extent=[0, 100, 0, 100], origin='upper', alpha=0.5, cmap='Blues')
    ax_.set_title(title, fontsize=6)
    ax_.axis('off')


def title_single_field(i, field):
    """
    Title generator for single fields.
    """
    return f'Cell {i}' \
           f'\nmean: {np.nanmean(field):.4f}; ' \
           f'{np.nanmean(np.abs(field)):.4f} (abs)' \
           f'\nmin: {np.nanmin(field):.4f} max: {np.nanmax(field):.4f}'


def print_dict(params):
    maxlen = max([len(s) for s in params.keys()])
    for k in params.keys():
        print(3*' ' + '| {}:{}{}'.format(k, (maxlen - len(k) + 1)*' ', params[k]))
    print()
