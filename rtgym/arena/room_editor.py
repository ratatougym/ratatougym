import numpy as np
import matplotlib.pyplot as plt

class RoomEditor:
    def __init__(self, width, height, save_on_close=True):
        self.width = width
        self.height = height
        self.grid = np.ones((height, width), dtype=int)  # Initialize maze grid
        self.save_on_close = save_on_close

        # State for drawing rectangles
        self.dragging = False
        self.start_point = None  # Starting point of the drag
        self.draw_value = 0  # Default to drawing white (0)

        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(self.grid, cmap='gray_r', vmin=0, vmax=1)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        self.shift_pressed = False  # Track if Shift is pressed

    def on_press(self, event):
        """Start dragging a rectangle."""
        if event.inaxes != self.ax:
            return
        self.dragging = True
        self.start_point = self.get_grid_coordinates(event)
        self.draw_value = 1 if self.shift_pressed else 0  # Set draw value based on Shift

    def on_drag(self, event):
        """Preview the rectangle while dragging."""
        if self.dragging and event.inaxes == self.ax:
            end_point = self.get_grid_coordinates(event)
            if self.start_point and end_point:
                self.preview_rectangle(self.start_point, end_point)

    def on_release(self, event):
        """Finish drawing the rectangle."""
        if self.dragging and self.start_point:
            end_point = self.get_grid_coordinates(event)
            if end_point:
                self.fill_rectangle(self.start_point, end_point)
        self.dragging = False
        self.start_point = None

    def on_key(self, event):
        """Handle key press events."""
        if event.key == 'shift':  # Detect if Shift is pressed
            self.shift_pressed = True
        elif event.key == 's':  # Save the grid as a file
            self.save_grid()
        elif event.key == 'q':  # Quit the editor
            plt.close(self.fig)

    def on_key_release(self, event):
        """Handle key release events."""
        if event.key == 'shift':  # Detect if Shift is released
            self.shift_pressed = False

    def on_close(self, event):
        """Save the grid when the window is closed."""
        if self.save_on_close:
            self.save_grid()

    def get_grid_coordinates(self, event):
        """Convert event coordinates to grid coordinates."""
        try:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= x < self.width and 0 <= y < self.height:
                return x, y
        except (TypeError, ValueError):
            pass
        return None

    def fill_rectangle(self, start_point, end_point):
        """Fill the rectangle defined by start and end points."""
        x0, y0 = start_point
        x1, y1 = end_point

        # Ensure the rectangle fills the correct region regardless of drag direction
        x_start, x_end = sorted((x0, x1))
        y_start, y_end = sorted((y0, y1))

        self.grid[y_start:y_end + 1, x_start:x_end + 1] = self.draw_value
        self.update_plot()

    def preview_rectangle(self, start_point, end_point):
        """Preview the rectangle during drag."""
        # Create a temporary copy of the grid for preview
        preview_grid = self.grid.copy()
        x0, y0 = start_point
        x1, y1 = end_point

        # Ensure the rectangle previews the correct region
        x_start, x_end = sorted((x0, x1))
        y_start, y_end = sorted((y0, y1))

        preview_grid[y_start:y_end + 1, x_start:x_end + 1] = self.draw_value
        self.im.set_data(preview_grid)
        self.fig.canvas.draw_idle()

    def save_grid(self):
        """Save the grid to a binary file."""
        filename = 'room.npy'
        np.save(filename, self.grid)
        print(f"Room saved as '{filename}'.")

    def update_plot(self):
        """Update the display."""
        self.im.set_data(self.grid)
        self.fig.canvas.draw_idle()

    def show(self):
        """Display the editor."""
        plt.title("Room Editor (Drag to draw rectangle, 'Shift' for black, 's' to save, 'q' to quit)")
        plt.show()


# Initialize and display the editor
editor = RoomEditor(width=30, height=60)
editor.show()
