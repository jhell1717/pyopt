import torch
from .func import _black_box_function

class Data:
    def __init__(self, low_lim, up_lim, num_points=5):
        self.low_lim = low_lim
        self.up_lim = up_lim
        self.x = self._sample_input(num_points)
        self.y = self._sample_input(num_points)
        self.z = self._evaluate(self.x,self.y)

    def _sample_input(self, num_points):
        """Sample input points from a grid instead of continuous space."""
        # Create a grid of possible values (same as visualization grid)
        grid_points = torch.linspace(self.low_lim, self.up_lim, 200)
        
        # Sample indices from the grid
        indices = torch.randint(0, 200, (num_points,))
        
        # Return the grid values at those indices
        return grid_points[indices].unsqueeze(1)

    def _evaluate(self, x,y):
        """Evaluate the black-box function on input x."""
        return _black_box_function(x,y)

    def create_vis_data(self, num_points=200):
        """Generate evenly spaced data for plotting the black-box function."""
        x_plot = torch.linspace(self.low_lim, self.up_lim, 200)
        y_plot = torch.linspace(self.low_lim, self.up_lim, 200)

        # Use 'ij' indexing to match the GP prediction grid
        x, y = torch.meshgrid(x_plot, y_plot, indexing='ij')
        z_plot = self._evaluate(x, y).detach().cpu().numpy()
        return x_plot, y_plot, z_plot


