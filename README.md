# Bayesian Optimisation Package

A lightweight and modular Python package for building Gaussian Process surrogate models and performing Bayesian Optimisation on an arbitary function. Ideal for optimisation tasks where function evaluations are expensive or noisy.

Unlike active learning where the objective is to learn as much about the underlying function in a finite number of evaluations as possible, Bayesian optimisation balances exploitation and exploration via smart acquisition functions to perform optimisation (e.g., minimisation or maximisation) In this example, we implement the upper confidence bound (UCB) that has a hyperparamater $$\beta$$ to control this balance.

## Features

- Build and train a Gaussian Process (GP) model using BoTorch
- Define and evaluate an 'unknown' function
- Optimise an acquisition function (UCB) to find promising candidate points
- Visualise the GP model and optimisation progress

---
## Examples

Example:
<a href="https://colab.research.google.com/github/jhell1717/pyopt/blob/dimensions%2F2D/examples/example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Quick Start

### 1. Install Dependencies

```bash
pip install torch botorch gpytorch matplotlib
