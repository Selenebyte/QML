import numpy as np


def get_scat_list(ax, parameters):
    """Create a list of scatters of the real part of the parameters.

    Creates a list of scatters, where the points are the real part of the parameters.

    Args:
        ax: Axes for the plot.
        parameters: Tensors of parameters.

    returns:
          List of scatters.

    Raises:
                    N/A
    """
    scat_list = []
    scat_name_list = []

    for j in range(parameters.shape[0]):
        scat = ax.scatter(*parameters[j], c="blue", label=f"Parameter {j + 1}")
        scat_name = ax.text(
            *parameters[j], "%s" % (str(j + 1)), size=20, zorder=1, color="k"
        )
        scat_list.append(scat)
        scat_name_list.append(scat_name)

    return (scat_list, scat_name_list)


def get_complex_scat_list(ax, parameters):
    """Create a list of scatters of the complex part of the parameters.

    Creates a list of scatters, where the points are the complex part of the parameters.

    Args:
        ax: Axes for the plot.
        parameters: Tensors of parameters.

    returns:
          List of scatters.

    Raises:
                    N/A
    """
    scat_list = []
    scat_name_list = []

    for j in range(parameters.shape[0]):
        scat = ax.scatter(
            np.real(parameters[j]),
            np.imag(parameters[j]),
            c="blue",
            label=f"Parameter {j + 1}",
        )
        scat_name = ax.text(
            np.real(parameters[j]),
            np.imag(parameters[j]),
            "%s" % (str(j + 1)),
            size=20,
            zorder=1,
            color="k",
        )
        scat_list.append(scat)
        scat_name_list.append(scat_name)

    return (scat_list, scat_name_list)
