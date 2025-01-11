import numpy as np

from util.math.normalize_vector import normalize_vector
from util.math.normalize_vector import normalize_complex_vector


def get_quiv_list(ax, parameters, parameters_grad):
    """Create a list of quivers of the real part of the gradient direction.

    Creates a list of quivers, where vectors with position at the real part of the parameters and points in the direction of the real part of the gradient.

    Args:
        ax: Axes for the plot.
        parameters: Tensors of parameters.
        parameters_grad: Tensors of parameter gradients.

    returns:
          List of quivers.

    Raises:
                    N/A
    """
    quiv_list = []

    for j in range(parameters.shape[0]):
        gradient_norm = normalize_vector(parameters_grad[j])
        quiv = ax.quiver(
            *parameters[j],
            *gradient_norm,
            color="red",
            label=f"Parameter {j + 1} Grad Vec",
        )
        quiv_list.append(quiv)

    return quiv_list


def get_complex_quiv_list(ax, parameters, parameters_grad):
    """Create a list of quivers of the complex part of the gradient direction.

    Creates a list of quivers, where vectors with position at the complex part of the parameters and points in the direction of the complex part of the gradient.

    Args:
        ax: Axes for the plot.
        parameters: Tensors of parameters.
        parameters_grad: Tensors of parameter gradients.

    returns:
          List of quivers.

    Raises:
                    N/A
    """
    parameters_grad = parameters_grad.cpu().numpy()

    quiv_list = []

    for j in range(parameters.shape[0]):
        gradient_norm = normalize_complex_vector(parameters_grad[j])
        quiv = ax.quiver(
            np.real(parameters[j]),
            np.imag(parameters[j]),
            np.real(gradient_norm),
            np.imag(gradient_norm),
            color="red",
            label=f"Parameter {j + 1} Grad Vec",
        )
        quiv_list.append(quiv)

    return quiv_list
