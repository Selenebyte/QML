from functools import partial

from abc import abstractmethod
from abc import ABC

import torch

import pennylane as qml


class model_framework(ABC, torch.nn.Module):
    @abstractmethod
    def __init__(self, model):
        super(model_framework, self).__init__()

        self.model = model

    @abstractmethod
    def Trainable_circuit_block(parameter):
        pass

    @abstractmethod
    def Data_encoding_curcuit_block(x):
        pass

    @abstractmethod
    def Quantum_model(self, parameters_, inputs):
        for parameter in parameters_[:-1]:
            self.Trainable_circuit_block(parameter)
            self.Data_encoding_curcuit_block(inputs)

        self.Trainable_circuit_block(parameters_[-1])
        return qml.expval(qml.PauliZ(wires=0))

    @abstractmethod
    def Measureable_quantum_model(
        self, parameters, inputs, dev, degree, noise_model=False
    ):
        def Quantum_model(parameters, inputs, final_layer=False):
            if final_layer:
                for parameter in parameters[:-1]:
                    self.Trainable_circuit_block(parameter)
                    self.Data_encoding_curcuit_block(inputs)

                self.Trainable_circuit_block(parameters[-1])
            else:
                for parameter in parameters:
                    self.Trainable_circuit_block(parameter)
                    self.Data_encoding_curcuit_block(inputs)

            return qml.expval(qml.PauliZ(wires=0))

        intermediate_states = []
        fourier_coef_list = []

        for i in range(1, parameters.shape[0] + 1):
            parameters_for_model = parameters[:i]

            q_node = qml.QNode(Quantum_model, dev)
            if noise_model:
                q_node = qml.add_noise(q_node, noise_model)

            is_final_layer = i == parameters.shape[0]

            fourier_coef = qml.fourier.coefficients(
                partial(q_node, parameters_for_model, final_layer=is_final_layer),
                1,
                degree,
            )
            intermediate_state = q_node(parameters_for_model, inputs, is_final_layer)

            intermediate_states.append(intermediate_state)
            fourier_coef_list.append(fourier_coef)

        return (intermediate_states, fourier_coef_list)

    @abstractmethod
    def forward(self, x):
        pass
