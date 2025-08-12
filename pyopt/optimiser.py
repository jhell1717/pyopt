import torch
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.optim import optimize_acqf
from .utils import plot_gp_optim


class Optimiser:
    def __init__(self, gp_model, q_samples=1, beta=0.1, raw_samples=20, acquisition_type="ucb"):
        self.gp_model = gp_model
        self.q_samples = q_samples
        self.beta = beta
        self.raw_samples = raw_samples
        self.acquisition_type = acquisition_type

        # Defer acquisition construction until optimise(), after ensuring GP is trained
        self.acq_func = None
        self.candidate = None
        self.new_z = None

    def _build_acquisition_function(self):
        """Create the specified acquisition function."""
        if self.acquisition_type.lower() == "ei":
            return ExpectedImprovement(self.gp_model.gp, best_f=self._get_best_f())
        else:  # default to UCB
            return UpperConfidenceBound(self.gp_model.gp, beta=self.beta)

    def _get_best_f(self):
        """Get the best observed function value so far."""
        return self.gp_model.data.z.max()

    def optimise(self):
        """Optimise the acquisition function and update the GP model with a new observation."""
        if self.gp_model.gp is None:
            self.gp_model.train_gp()
        # Always rebuild the acquisition on the latest trained GP
        self.acq_func = self._build_acquisition_function()

        self.candidate, _ = optimize_acqf(
            acq_function=self.acq_func,
            bounds=self.gp_model.bounds,
            q=self.q_samples,
            num_restarts=5,
            raw_samples=self.raw_samples,
        )
        self.new_z = self._evaluate_candidate(self.candidate)
        self._update_data(self.candidate, self.new_z.view(1,1))
        # Retrain GP on augmented dataset so the next iteration uses updated posterior
        self.gp_model.input, self.gp_model.output = self.gp_model._create_features()

    def _evaluate_candidate(self, candidate):
        """Evaluate the candidate point using the black-box function."""
        return self.gp_model.data._evaluate(candidate[0][0],candidate[0][1])

    def _update_data(self, candidate, z):
        """Append the new observation to the GP model’s data."""
        data = self.gp_model.data
        x,y = torch.split(candidate,1,dim=1)
        data.x = torch.cat([data.x, x])
        data.y = torch.cat([data.y, y])
        data.z = torch.cat([data.z, z])

    def visualise(self):
        """Plot the GP model and the newly selected candidate point."""
        x_plot, y_plot, z_plot = self.gp_model.data.create_vis_data()
        X,Y = torch.meshgrid(x_plot,y_plot,indexing='ij')
        X_Y_flat = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

        _, mean, std = self.gp_model.get_posterior(X_Y_flat)

        mean = mean.reshape(200,200)
        std = std.reshape(200,200)

        plot_gp_optim(x_plot,y_plot,z_plot,mean.reshape(200,200),std.reshape(200,200),self.gp_model.data)

        return x_plot,y_plot,z_plot,mean,std

