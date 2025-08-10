import torch

# Define the black-box function (1D example)
def black_box_function(x):
    return torch.sin(3 * x) + 0.3 * torch.cos(10 * x)

# Define a more complex black-box function
def complex_black_box_function(x):
    # Ensure x is a torch tensor
    x = x.clone()
    
    # Base periodic component with multiple harmonics
    term1 = torch.sin(5 * x) * torch.cos(3 * x) + 0.5 * torch.sin(15 * x)

    # Localized peak using Gaussian bump
    term2 = torch.exp(-((x - 1.5) ** 2) / 0.01)

    # Discontinuous jump
    term3 = torch.where(x > 2.0, torch.tensor(2.0, dtype=x.dtype), torch.tensor(0.0, dtype=x.dtype))

    # High frequency burst within a specific interval
    term4 = torch.where((x > 0.5) & (x < 0.7), torch.sin(50 * x), torch.tensor(0.0, dtype=x.dtype))

    # Combine all terms
    return term1 + term2 + term3 + term4