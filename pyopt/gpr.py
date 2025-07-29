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
        self.gp = None

    def _get_device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_bounds(self):
        return torch.tensor([[self.data.low_lim], [self.data.up_lim]], dtype=torch.float32)

    def train_gp(self):
        """Train the Gaussian Process model using the data."""
        self.gp = SingleTaskGP(self.data.x, self.data.y)
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
        x_plot, y_plot = self.data.create_vis_data()
        _, mean, std = self.get_posterior(x_plot)
        plot_gp_plain(x_plot, y_plot, mean, std, self.data)
    

