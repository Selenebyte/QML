import matplotlib.animation as animation

from util.visualization.scatter_list import get_scat_list
from util.visualization.scatter_list import get_complex_scat_list
from util.visualization.quiver_list import get_quiv_list
from util.visualization.quiver_list import get_complex_quiv_list


def plot_parameters(fig, ax, parameters_list, parameters_grad_list):
    """Get an animation of the real part of the parameters with gradient over steps.

    Get an animation of the real part of the parameters with a red arrow that shows the gradient of all the list elements.

    Args:
                    fig: Figure for the plot.
        ax: Axes for the plot.
        parameters_list: List of tensors of parameters.
        parameters_grad_list: List of tensors of parameter gradients.

    returns:
          Animation.

    Raises:
                    N/A
    """

    def animate(i):
        nonlocal scat_list
        nonlocal scat_name_list
        nonlocal quiv_list

        for scat in scat_list:
            scat.remove()

        for scat_name in scat_name_list:
            scat_name.remove()

        (scat_list, scat_name_list) = get_scat_list(ax, parameters_list[i])

        if not i == len(parameters_list) - 1:
            for quiv in quiv_list:
                quiv.remove()

            quiv_list = get_quiv_list(ax, parameters_list[i], parameters_grad_list[i])
            return (
                scat_list,
                scat_name_list,
                quiv_list,
            )

        return (
            scat_list,
            scat_name_list,
        )

    (scat_list, scat_name_list) = get_scat_list(ax, parameters_list[0])
    quiv_list = get_quiv_list(ax, parameters_list[0], parameters_grad_list[0])

    ani = animation.FuncAnimation(
        fig, animate, interval=50, blit=False, save_count=len(parameters_list)
    )

    return ani


def plot_complex_parameters(fig, ax, parameters_list, parameters_grad_list):
    """Get an animation of the complex part of the parameters with gradient over steps.

    Get an animation of the complex part of the parameters with a red arrow that shows the gradient of all the list elements.

    Args:
                    fig: Figure for the plot.
        ax: Axes for the plot.
        parameters_list: List of tensors of parameters.
        parameters_grad_list: List of tensors of parameter gradients.

    returns:
          Animation.

    Raises:
                    N/A
    """

    def animate(i):
        nonlocal scat_list
        nonlocal scat_name_list
        nonlocal quiv_list

        for scat in scat_list:
            scat.remove()

        for scat_name in scat_name_list:
            scat_name.remove()

        (scat_list, scat_name_list) = get_complex_scat_list(ax, parameters_list[i])

        if not i == len(parameters_list) - 1:
            for quiv in quiv_list:
                quiv.remove()

            quiv_list = get_complex_quiv_list(
                ax, parameters_list[i], parameters_grad_list[i]
            )
            return (
                scat_list,
                scat_name_list,
                quiv_list,
            )

        return (
            scat_list,
            scat_name_list,
        )

    (scat_list, scat_name_list) = get_complex_scat_list(ax, parameters_list[0])
    quiv_list = get_complex_quiv_list(ax, parameters_list[0], parameters_grad_list[0])

    ani = animation.FuncAnimation(
        fig, animate, interval=50, blit=False, save_count=len(parameters_list)
    )

    return ani
