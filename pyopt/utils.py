import matplotlib.pyplot as plt

def plot_gp_plain(x,y,mean,std,data):
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
    # if candidate is not None:
    #     plt.scatter(candidate.numpy(), new_y.numpy(),
    #                 c='g', marker='*', s=200, label='New Point')
    plt.legend()
    plt.title("Bayesian Optimisation")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

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
