import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_gp_plain(X, Y, Z, mean, std, data):
    fig = go.Figure()

    # Mean surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        opacity=0.8,
        name='GP Mean'
    ))
    

    fig.update_layout(
        title="Bayesian Optimisation (3D GP Surface)",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x, y)"
        ),
        scene_aspectmode='cube',
        )
    
    fig.add_trace(go.Scatter3d(
        x=data.x.squeeze(),
        y=data.y.squeeze(),
        z=data.z.squeeze(),
        mode='markers',
        marker=dict(size=6, color='red', symbol='circle'),
        name='Sampled Points'
        ))
    fig.show()

def plot_gp_optim(x,y,mean,std,data,candidate,new_y):
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x.numpy(), y, 'k--', label='True Function')
    plt.plot(x.numpy(), mean, 'b', label='GP Mean')
    plt.fill_between(
        x.squeeze().numpy(),
        (mean - 2 * std).squeeze(),
        (mean + 2 * std).squeeze(),
        alpha=0.3,
        label='Uncertainty (2 std)',
    )
    plt.scatter(data.x.numpy(),
                data.y.numpy(), c='r', label='Sampled Points')
   
    plt.scatter(candidate.numpy(), new_y.numpy(),
                    c='g', marker='*', s=200, label='New Point')
    plt.legend()
    plt.title("Bayesian Optimisation")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()
