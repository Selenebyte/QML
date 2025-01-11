import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import numpy as np

import pennylane as qml

from models.non_noise_models.model import quantum_model
from models.noise_models.noise_model import quantum_noise_model

from util.training.training_objectives import target_function
from util.metrics.correlations_intermediate_states import (
    get_correlations_between_intermediate_states,
)

from models.train import train


def get_average_correlation(
    loops,
    validation_x,
    fourier_coeffs,
    parameter_shape,
    degree,
    layers,
    device,
    noise_model=None,
):
    # Losses
    list_of_losses = []

    # Pearson Correlation
    list_of_pearson_correlations_loops = []
    list_of_real_fourier_pearson_correlations_loops = []
    list_of_complex_fourier_pearson_correlations_loops = []

    # Distance Correlation
    list_of_distance_correlations_loops = []
    list_of_real_fourier_distance_correlations_loops = []
    list_of_complex_fourier_distance_correlations_loops = []

    # Mutual Info
    list_of_mutual_info_loops = []
    list_of_real_fourier_mutual_info_loops = []
    list_of_complex_fourier_mutual_info_loops = []

    # Cross Correlation
    list_of_cross_correlations_loops = []
    list_of_real_fourier_cross_correlations_loops = []
    list_of_complex_fourier_cross_correlations_loops = []

    for i in range(loops):
        print(f"loop: {i}")
        x = torch.linspace(-10, 10, 100, requires_grad=False).to(device)

        if noise_model:
            dev = qml.device("default.mixed", wires=1)
            model = quantum_noise_model(parameter_shape, dev, noise_model)
        else:
            dev = qml.device("default.qubit", wires=1)
            model = quantum_model(parameter_shape, dev)

        validation_y = target_function(x, fourier_coeffs, device)
        validation_y = validation_y / (
            torch.ceil(torch.max(validation_y)).to(torch.int)
        )
        target_y = validation_y

        epochs = 100

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        (_, parameters_list, _, _, _) = train(
            model,
            x,
            target_y,
            optimizer,
            loss_fn,
            layers,
            epochs,
            dev,
            device,
            noise_model,
        )

        if noise_model:
            (intermediate_states, fourier_coef_list) = model.Measureable_quantum_model(
                parameters_list[-1], validation_x, noise_model, dev, degree
            )
        else:
            (intermediate_states, fourier_coef_list) = model.Measureable_quantum_model(
                parameters_list[-1], validation_x, dev, degree
            )

        intermediate_states = [validation_x] + intermediate_states

        (
            list_of_pearson_correlations,
            list_of_real_fourier_pearson_correlations,
            list_of_complex_fourier_pearson_correlations,
        ) = get_correlations_between_intermediate_states(
            intermediate_states, fourier_coef_list, "pearson_correlation"
        )
        (
            list_of_distance_correlations,
            list_of_real_fourier_distance_correlations,
            list_of_complex_fourier_distance_correlations,
        ) = get_correlations_between_intermediate_states(
            intermediate_states, fourier_coef_list, "distance_correlation"
        )
        (
            list_of_mutual_info,
            list_of_real_fourier_mutual_info,
            list_of_complex_fourier_mutual_info,
        ) = get_correlations_between_intermediate_states(
            intermediate_states, fourier_coef_list, "mutual_info"
        )
        (
            list_of_cross_correlations,
            list_of_real_fourier_cross_correlations,
            list_of_complex_fourier_cross_correlations,
        ) = get_correlations_between_intermediate_states(
            intermediate_states, fourier_coef_list, "cross_correlation"
        )

        # Losses
        validation_y = target_function(validation_x, fourier_coeffs, device)
        validation_y = validation_y / (
            torch.ceil(torch.max(validation_y)).to(torch.int)
        )
        validation_loss = loss_fn(model(validation_x), validation_y)
        list_of_losses.append(validation_loss)

        # Pearson Correlation
        list_of_pearson_correlations_loops.append(list_of_pearson_correlations)
        list_of_real_fourier_pearson_correlations_loops.append(
            list_of_real_fourier_pearson_correlations
        )
        list_of_complex_fourier_pearson_correlations_loops.append(
            list_of_complex_fourier_pearson_correlations
        )

        # Distance Correlation
        list_of_distance_correlations_loops.append(list_of_distance_correlations)
        list_of_real_fourier_distance_correlations_loops.append(
            list_of_real_fourier_distance_correlations
        )
        list_of_complex_fourier_distance_correlations_loops.append(
            list_of_complex_fourier_distance_correlations
        )

        # Mutual Cnfo
        list_of_mutual_info_loops.append(list_of_mutual_info)
        list_of_real_fourier_mutual_info_loops.append(list_of_real_fourier_mutual_info)
        list_of_complex_fourier_mutual_info_loops.append(
            list_of_complex_fourier_mutual_info
        )

        # Cross Correlation
        list_of_cross_correlations_loops.append(list_of_cross_correlations)
        list_of_real_fourier_cross_correlations_loops.append(
            list_of_real_fourier_cross_correlations
        )
        list_of_complex_fourier_cross_correlations_loops.append(
            list_of_complex_fourier_cross_correlations
        )

    # Losses
    mean_losses = torch.mean(torch.stack(list_of_losses), axis=0)

    # Pearson Correlation
    mean_pearson_correlations = torch.mean(
        torch.as_tensor(list_of_pearson_correlations_loops), axis=0
    )
    mean_real_fourier_pearson_correlations = torch.mean(
        torch.as_tensor(list_of_real_fourier_pearson_correlations_loops), axis=0
    )
    mean_imag_fourier_pearson_correlations = torch.mean(
        torch.as_tensor(list_of_complex_fourier_pearson_correlations_loops), axis=0
    )

    # Distance Correlation
    mean_distance_correlations = torch.mean(
        torch.as_tensor(list_of_distance_correlations_loops), axis=0
    )
    mean_real_fourier_distance_correlations = torch.mean(
        torch.as_tensor(list_of_real_fourier_distance_correlations_loops), axis=0
    )
    mean_imag_fourier_distance_correlations = torch.mean(
        torch.as_tensor(list_of_complex_fourier_distance_correlations_loops), axis=0
    )

    # Mutual Info
    mean_mutual_info = torch.mean(
        torch.as_tensor(np.array(list_of_mutual_info_loops)), axis=0
    )
    mean_mutual_info = mean_mutual_info.reshape(
        (mean_mutual_info.shape[0], mean_mutual_info.shape[0])
    )

    mean_real_fourier_mutual_info = torch.mean(
        torch.as_tensor(np.array(list_of_real_fourier_mutual_info_loops)), axis=0
    )
    mean_real_fourier_mutual_info = mean_real_fourier_mutual_info.reshape(
        (mean_real_fourier_mutual_info.shape[0], mean_real_fourier_mutual_info.shape[0])
    )

    mean_complex_fourier_mutual_info = torch.mean(
        torch.as_tensor(np.array(list_of_complex_fourier_mutual_info_loops)), axis=0
    )
    mean_complex_fourier_mutual_info = mean_complex_fourier_mutual_info.reshape(
        (
            mean_complex_fourier_mutual_info.shape[0],
            mean_complex_fourier_mutual_info.shape[0],
        )
    )

    # Cross Correlation
    mean_cross_correlations = torch.mean(
        torch.as_tensor(np.array(list_of_cross_correlations_loops)), axis=0
    )
    mean_cross_correlations = torch.max(mean_cross_correlations, axis=-1)[0]

    mean_real_fourier_cross_correlations = torch.mean(
        torch.as_tensor(np.array(list_of_real_fourier_cross_correlations_loops)), axis=0
    )
    mean_real_fourier_cross_correlations = torch.max(
        mean_real_fourier_cross_correlations, axis=-1
    )[0]

    mean_complex_fourier_cross_correlations = torch.mean(
        torch.as_tensor(np.array(list_of_complex_fourier_cross_correlations_loops)),
        axis=0,
    )
    mean_complex_fourier_cross_correlations = torch.max(
        mean_complex_fourier_cross_correlations, axis=-1
    )[0]

    return (
        mean_losses,
        mean_pearson_correlations,
        mean_real_fourier_pearson_correlations,
        mean_imag_fourier_pearson_correlations,
        mean_distance_correlations,
        mean_real_fourier_distance_correlations,
        mean_imag_fourier_distance_correlations,
        mean_mutual_info,
        mean_real_fourier_mutual_info,
        mean_complex_fourier_mutual_info,
        mean_cross_correlations,
        mean_real_fourier_cross_correlations,
        mean_complex_fourier_cross_correlations,
    )
