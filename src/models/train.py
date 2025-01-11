import pennylane as qml

import torch

from models.non_noise_models.model import quantum_model

from util.training.training_objectives import loss_of_fourier_coefficients

from functools import partial


def train(
    model,
    x: torch.Tensor,
    target_y: torch.Tensor,
    optimizer,
    loss_fn,
    degree: int,
    epochs: int,
    dev,
    device,
    noise_model=None,
):
    loss_list = [loss_fn(model(x), target_y)]

    parameters_list = [torch.clone(next(model.parameters())).detach()]
    parameters_grad_list = []

    q_node = qml.QNode(model.Quantum_model, dev, interface="torch")
    if noise_model:
        q_node = qml.add_noise(q_node, noise_model)

    fourier_coef_list = [
        qml.fourier.coefficients(partial(q_node, next(model.parameters())), 1, degree)
    ]
    fourier_coef_grad_list = []

    model.train()

    for epoch in range(epochs):
        x_data = x
        y_data = target_y
        optimizer.zero_grad()
        loss = loss_fn(model(x_data), y_data)
        loss.backward()
        optimizer.step()

        loss = loss_fn(model(x), target_y)

        loss_list.append(loss)

        parameters_list.append(torch.clone(next(model.parameters())).detach())
        parameters_grad_list.append(next(model.parameters()).grad)

        fourier_coef = qml.fourier.coefficients(
            partial(q_node, next(model.parameters())), 1, degree
        )
        fourier_coef_list.append(fourier_coef)

        fourier_coef = torch.tensor(fourier_coef).to(device).requires_grad_()
        loss_of_fourier_coef = loss_of_fourier_coefficients(
            fourier_coef, x, target_y, loss_fn, device
        )
        loss_of_fourier_coef.backward()

        fourier_coef_grad_list.append(fourier_coef.grad)

        # if (epoch + 1) % 10 == 0:
        #  print("Cost at step {0:3}: {1}".format(epoch + 1, loss))

    return (
        loss_list,
        parameters_list,
        parameters_grad_list,
        fourier_coef_list,
        fourier_coef_grad_list,
    )
