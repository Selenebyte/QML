import torch
from torchmetrics import PearsonCorrCoef

from scipy import signal

from sklearn.feature_selection import mutual_info_regression

import dcor


def get_correlations_between_intermediate_states(
    intermediate_states, fourier_coef_list, metric
):
    """Correlations between intermediate states and fourier coefficients.

    Computes the correlation between all pair permutations of intermediate states and their fourier coefficients.
    Correlations currently supported are distance, cross, and pearson correlations and mutual information.

    Args:
                    intermediate_states: List of intermediate states.
        fourier_coef_list: List of fourier coefficients.
        metric: Correlation metric to use. Supports "Distance_correlation", "mutual_info", and "cross_correlation". Defaults to pearson correlation.

    returns:
          Returns three lists of lists of intermediate states, and real and complex fourier coefficients.
        The first list corresponds to the current intermediate state being considered.
        The second list corresponds to the correlation between the current intermediate state and all the other intermediate states including itself.

    Raises:
                    N/A
    """
    if metric == "distance_correlation":
        metric_function = lambda x1, x2: dcor.distance_correlation(x1, x2, exponent=0.5)
    elif metric == "mutual_info":
        metric_function = lambda x1, x2: mutual_info_regression(
            x1.cpu().numpy().reshape(-1, 1), x2.cpu().numpy(), n_neighbors=2
        )
    elif metric == "cross_correlation":
        metric_function = lambda x1, x2: signal.correlate(
            x1.cpu().numpy(), x2.cpu().numpy()
        )
    else:
        pearson = PearsonCorrCoef().to(torch.float64)
        metric_function = lambda x1, x2: pearson(
            x1.to(torch.float64), x2.to(torch.float64)
        )

    list_of_correlations = []
    list_of_real_fourier_correlations = []
    list_of_complex_fourier_correlations = []
    for i in range(len(intermediate_states)):
        list_of_correlations_of_intermediate_state = []
        for j in range(len(intermediate_states)):
            list_of_correlations_of_intermediate_state.append(
                metric_function(
                    intermediate_states[j].cpu(), intermediate_states[i].cpu()
                )
            )

        list_of_correlations.append(list_of_correlations_of_intermediate_state)

    for i in range(len(fourier_coef_list)):
        list_of_real_fourier_correlations_of_intermediate_state = []
        list_of_complex_fourier_correlations_of_intermediate_state = []
        for j in range(len(fourier_coef_list)):
            list_of_real_fourier_correlations_of_intermediate_state.append(
                metric_function(
                    torch.real(torch.as_tensor(fourier_coef_list[j])),
                    torch.real(torch.as_tensor(fourier_coef_list[i])),
                )
            )
            list_of_complex_fourier_correlations_of_intermediate_state.append(
                metric_function(
                    torch.imag(torch.as_tensor(fourier_coef_list[j])),
                    torch.imag(torch.as_tensor(fourier_coef_list[i])),
                )
            )

        list_of_real_fourier_correlations.append(
            list_of_real_fourier_correlations_of_intermediate_state
        )
        list_of_complex_fourier_correlations.append(
            list_of_complex_fourier_correlations_of_intermediate_state
        )

    return (
        list_of_correlations,
        list_of_real_fourier_correlations,
        list_of_complex_fourier_correlations,
    )
