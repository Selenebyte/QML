from functools import partial

import contextlib

import torch

import numpy as np

import pennylane as qml

from models.model_framework import model_framework


class quantum_noise_model(model_framework):
    def __init__(self, parameter_shape, dev, noise_model):
        self.dev = dev

        q_node = qml.QNode(self.Quantum_model, self.dev)

        q_node = qml.add_noise(q_node, noise_model)

        q_layer = qml.qnn.TorchLayer(q_node, parameter_shape)

        model = torch.nn.Sequential(*[q_layer])

        super().__init__(model)

    @staticmethod
    def Data_encoding_curcuit_block(x):
        qml.RX(x, wires=0)

    @staticmethod
    def Trainable_circuit_block(parameter):
        qml.Rot(phi=parameter[0], theta=parameter[1], omega=parameter[2], wires=0)

    def Quantum_model(self, parameters_, inputs):
        return super().Quantum_model(parameters_, inputs)

    def Measureable_quantum_model(self, parameters, inputs, noise_model, dev, degree):
        return super().Measureable_quantum_model(
            parameters, inputs, dev, degree, noise_model
        )

    def forward(self, x):
        predictions = self.model(x)

        return predictions
