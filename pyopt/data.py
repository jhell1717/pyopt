import torch
from .func import black_box_function

class Data:
    def __init__(self, low_lim, up_lim, num_points=5):
        self.low_lim = low_lim
        self.up_lim = up_lim
        self.x = self._sample_input(num_points)
        self.y = self._evaluate(self.x)

    def _sample_input(self, num_points):
        """Sample input points uniformly in [0, 1]."""
        # return torch.rand(num_points, 1)
        return self.low_lim + (self.up_lim - self.low_lim) * torch.rand(num_points, 1)

    def _evaluate(self, x):
        """Evaluate the black-box function on input x."""
        return black_box_function(x)

    def create_vis_data(self, num_points=200):
        """Generate evenly spaced data for plotting the black-box function."""
        x_plot = torch.linspace(self.low_lim, self.up_lim, num_points).unsqueeze(1)
        y_plot = self._evaluate(x_plot).detach().cpu().numpy()
        return x_plot, y_plot


