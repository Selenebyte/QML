import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from util.training.training_objectives import predictions_from_target_function

from util.visualization.animation_plots import plot_parameters
from util.visualization.animation_plots import plot_complex_parameters


def visualize_curve_under_training(
    x,
    target_y_no_noise,
    fourier_coef_list,
    device,
    filename=None,
    reverse_conj_coef=False,
):
    """Creates an animation of the fourier function.

    Creates an animation of the fourier coefficient list and shows the plot of the target output.
    If a filename is provided then save the file as "{filename}.gif".

    Args:
        x: Input data.
        target_y_no_noise: Target output.
        fourier_coef_list: List of the fourier coefficients that are animated.
        device: Device for the torch tensors.
        filename: Filename. Default: None.

    returns:
          Animation of the fourier list.

    Raises:
                    N/A
    """
    y_predictions_for_coefficients = predictions_from_target_function(
        x, fourier_coef_list, device, reverse_conj_coef
    )

    fig, ax = plt.subplots()

    plt.plot(x.cpu(), target_y_no_noise.cpu(), c="black", label="Target Curve")
    (line,) = ax.plot(
        x.cpu(),
        y_predictions_for_coefficients[0].cpu(),
        c="blue",
        label="Predicted Curve",
    )

    def animate(i):
        line.set_ydata(y_predictions_for_coefficients[i].cpu())
        return (line,)

    ani = animation.FuncAnimation(
        fig,
        animate,
        interval=50,
        blit=True,
        save_count=len(y_predictions_for_coefficients),
    )

    plt.ylim(-1, 1)

    plt.title("")
    plt.ylabel("y")
    plt.xlabel("x")

    ax.legend()

    plt.close()

    if filename:
        ani.save(f"{filename}.gif", dpi=300, writer=animation.PillowWriter(fps=30))

    return ani


def visualize_real_part_of_parameters_under_training(
    parameters_list, parameters_grad_list, filename=None
):
    """Creates an animation of the real part of the parameters.

    Creates an animation of the real part of the parameter list with the real part of the gradient direction.
    If a filename is provided then save the file as "{filename}.gif".

    Args:
        parameters_list: List of tensors of parameters.
        parameters_grad_list: List of tensors of parameter gradients.
        filename: Filename. Default: None.

    returns:
          Animation of the parameters.

    Raises:
                    N/A
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ani = plot_parameters(fig, ax, parameters_list, parameters_grad_list)

    ax.set_xlabel("Phi")
    ax.set_ylabel("Theta")
    ax.set_zlabel("Omega")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_zlim(0, 5)

    plt.close()

    if filename:
        ani.save(f"{filename}.gif", dpi=300, writer=animation.PillowWriter(fps=30))

    return ani


def visualize_complex_part_of_parameters_under_training(
    parameters_list, parameters_grad_list, filename=None
):
    """Creates an animation of the complex part of the parameters.

    Creates an animation of the complex part of the parameter list with the complex part of the gradient direction.
    If a filename is provided then save the file as "{filename}.gif".

    Args:
        parameters_list: List of tensors of parameters.
        parameters_grad_list: List of tensors of parameter gradients.
        filename: Filename. Default: None.

    returns:
          Animation of the parameters.

    Raises:
                    N/A
    """
    fig = plt.figure()
    ax = fig.add_subplot()

    ani = plot_complex_parameters(fig, ax, parameters_list, parameters_grad_list)

    ax.set_xlabel("Real")
    ax.set_ylabel("Complex")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    plt.grid()
    plt.close()

    if filename:
        ani.save(f"{filename}.gif", dpi=300, writer=animation.PillowWriter(fps=30))

    return ani


def visualize_prediction_and_target_curve(x, target_y_no_noise, model):
    """Visualize model output with target function.

    Visualize model output with target function.

    Args:
        x: Input data.
        target_y_no_noise: Target output.
        model: Model.

    returns:
          Figure of the plot.

    Raises:
                    N/A
    """
    predictions = model(x).cpu().detach()

    fig, ax = plt.subplots()

    ax.plot(x.cpu(), target_y_no_noise.cpu(), c="black", label="Target Curve")
    ax.scatter(
        x.cpu(),
        target_y_no_noise.cpu(),
        facecolor="white",
        edgecolor="black",
        label="Target Training Points",
    )
    ax.plot(x.cpu(), predictions, c="blue", label="Prediction Curve")
    plt.ylim(-1, 1)

    plt.title("")
    plt.ylabel("y")
    plt.xlabel("x")

    ax.legend()

    return fig


def visualize_loss_under_training(loss_list):
    """Visualize loss under training.

    Visualize loss under training as a function of the epoch.

    Args:
        loss_list: List of loss during training.

    returns:
          Figure of the plot.

    Raises:
                    N/A
    """
    fig, ax = plt.subplots()

    ax.plot(range(len(loss_list)), torch.as_tensor(loss_list).cpu(), label="Loss")

    plt.title("Loss at each epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    ax.legend()

    return fig
