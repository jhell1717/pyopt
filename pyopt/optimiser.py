import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from .utils import plot_gp_optim


class Optimiser:
    def __init__(self, gp_model, q_samples=1, beta=0.1, raw_samples=20):
        self.gp_model = gp_model
        self.q_samples = q_samples
        self.beta = beta
        self.raw_samples = raw_samples

        self.acq_func = self._build_acquisition_function()
        self.candidate = None
        self.new_y = None

    def _build_acquisition_function(self):
        """Create an Upper Confidence Bound acquisition function."""
        return UpperConfidenceBound(self.gp_model.gp, beta=self.beta)

    def optimise(self):
        """Optimise the acquisition function and update the GP model with a new observation."""
        self.candidate, _ = optimize_acqf(
            acq_function=self.acq_func,
            bounds=self.gp_model.bounds,
            q=self.q_samples,
            num_restarts=5,
            raw_samples=self.raw_samples,
        )
        self.new_y = self._evaluate_candidate(self.candidate)
        self._update_data(self.candidate, self.new_y)

    def _evaluate_candidate(self, x):
        """Evaluate the candidate point using the black-box function."""
        return self.gp_model.data._evaluate(x)

    def _update_data(self, x, y):
        """Append the new observation to the GP model’s data."""
        data = self.gp_model.data
        data.x = torch.cat([data.x, x])
        data.y = torch.cat([data.y, y])

    def visualise(self):
        """Plot the GP model and the newly selected candidate point."""
        x_plot, y_plot = self.gp_model.data.create_vis_data()
        _, mean, std = self.gp_model.get_posterior(x_plot)

        plot_gp_optim(
            x=x_plot,
            y=y_plot,
            mean=mean,
            std=std,
            data=self.gp_model.data,
            candidate=self.candidate,
            new_y=self.new_y,
        )
