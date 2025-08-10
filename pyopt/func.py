import torch

# Define the black-box function (1D example)
def black_box_function(x):
    return torch.sin(3 * x) + 0.3 * torch.cos(10 * x)

def _black_box_function(x,y):
    return torch.sin(3 * x) + 0.3 * torch.cos(10 * x) + torch.sin(3 * y) + 0.3 * torch.cos(10 * y)
