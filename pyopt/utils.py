import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np 
import torch


import torch
import plotly.graph_objects as go

def plot_gp_plain(X, Y, Z, mean, std, data):
    fig = go.Figure()

    # Meshgrid for the plot
    X_mesh, Y_mesh = torch.meshgrid(X, Y, indexing='ij')

    # Mean surface coloured by std
    fig.add_trace(go.Surface(
        x=X_mesh.numpy(),
        y=Y_mesh.numpy(), 
        z=mean,                       # height = GP mean
        surfacecolor=std,              # colour = GP std
        colorscale='Viridis',
        opacity=0.9,
        name='GP Mean (coloured by std)',
        colorbar=dict(
        title='Std Dev',
        x=1.05,    # push colorbar further right
        y=0.5,
        len=0.75
    ),
        showlegend=True,
        contours=dict(
        x=dict(show=True, color='black', width=2),  # grid lines in x direction
        y=dict(show=True, color='black', width=2)   # grid lines in y direction
    )
        
    ))

    # True surface (light overlay)
    fig.add_trace(go.Surface(
        x=X_mesh.numpy(),
        y=Y_mesh.numpy(), 
        z=Z,
        colorscale='Reds',
        opacity=0.2,
        showscale=False,
        name='True Surface',
        showlegend=True,
    ))

    # Sampled points
    fig.add_trace(go.Scatter3d(
        x=data.x.squeeze(),
        y=data.y.squeeze(),
        z=data.z.squeeze(),
        mode='markers',
        marker=dict(size=6, color='black', symbol='circle'),
        name='Sampled Points'
    ))

    fig.update_layout(
        title="Bayesian Optimisation (3D GP Surface coloured by Std Dev)",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x, y)"
        ),
        scene_aspectmode='cube',
        legend=dict(
            x=0.02,  # move legend to top-left
            y=0.98,
            bgcolor='rgba(255,255,255,0.6)'
        )
    )


    fig.show()



def plot_gp_optim(X, Y, Z, mean, std, data):
    fig = go.Figure()

    # Create proper meshgrid for surface plotting
    X_mesh, Y_mesh = torch.meshgrid(X, Y, indexing='ij')

    fig.add_trace(go.Surface(
        x=X_mesh.numpy(),
        y=Y_mesh.numpy(),
        z=mean,
        surfacecolor=std,      # Color comes from uncertainty
        colorscale='Inferno',
        opacity=0.9,
        name='GP Mean (coloured by Std Dev)',
        showscale=True,
        colorbar=dict(title="Std Dev")
    ))

    # # Actual function surface
    # fig.add_trace(go.Surface(
    #     x=X_mesh.numpy(),
    #     y=Y_mesh.numpy(),
    #     z=Z,
    #     colorscale='Viridis',
    #     opacity=0.8,
    #     name='Actual Function'
    # ))


    
    fig.add_trace(go.Scatter3d(
        x=data.x.squeeze(),
        y=data.y.squeeze(),
        z=data.z.squeeze(),
        mode='markers',
        marker=dict(size=6, color='red', symbol='circle'),
        name='Sampled Points'
        ))
    
    fig.update_layout(
        title="Bayesian Optimisation (3D GP Surface + Uncertainty)",
        legend=dict(
            itemsizing='constant',
            bgcolor='rgba(255,255,255,0.7)'
        ),
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x, y)"
        ),
        scene_aspectmode='cube'
    )
    fig.show()
