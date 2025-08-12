import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from .utils import plot_gp_plain


class GP:
    def __init__(self, data):
        self.data = data
        self.device = self._get_device()
        self.bounds = self._create_bounds()
        self.input, self.output = self._create_features()
        self.gp = None

    def _get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _create_features(self):
        return torch.cat([self.data.x,self.data.y],dim=-1), self.data.z

    def _create_bounds(self):
        return torch.tensor([[self.data.low_lim,self.data.low_lim], [self.data.up_lim,self.data.up_lim]], dtype=torch.float32)

    def train_gp(self):
        """Train the Gaussian Process model using the data."""
        self.gp = SingleTaskGP(self.input, self.output)
        mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        fit_gpytorch_mll(mll)


    def get_posterior(self, x_input):
        """Compute GP posterior and return posterior object, mean, and std."""
        self.gp.eval()
        posterior = self.gp.posterior(x_input)
        return posterior, *self._extract_mean_std(posterior)

    def _extract_mean_std(self, posterior):
        mean = posterior.mean.detach().cpu().numpy()
        std = posterior.variance.sqrt().detach().cpu().numpy()
        return mean, std

    def plot_gp(self):
        """Visualize the GP predictions using the provided data object."""
        x_plot, y_plot, z_plot = self.data.create_vis_data()

        # Create meshgrid for visualization - use 'ij' indexing for consistency
        X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
        
        # Create flat coordinates for GP prediction - ensure proper ordering
        # This creates a flattened array that matches the meshgrid exactly
        x_coords = x_plot.unsqueeze(1).repeat(1, len(y_plot)).flatten()
        y_coords = y_plot.repeat(len(x_plot))
        X_Y_flat = torch.stack([x_coords, y_coords], dim=-1)

        _, mean, std = self.get_posterior(X_Y_flat)

        # Reshape predictions to match the grid dimensions
        # Note: We need to transpose because meshgrid with 'ij' indexing gives different shape
        mean = mean.reshape(len(x_plot), len(y_plot))
        std = std.reshape(len(x_plot), len(y_plot))
        
        plot_gp_plain(x_plot, y_plot, z_plot, mean, std, self.data)
        return x_plot, y_plot, z_plot, X_Y_flat