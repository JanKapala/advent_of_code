# pylint: disable=missing-module-docstring, invalid-name, too-many-arguments, no-member, fixme

import torch
from torch import Tensor
from torch.nn import Module, ReLU, Sequential

from day19.rl.agent.base_neural_network import BaseNeuralNetwork

# TODO: generally take care of fixme offences in the whole project (maybe exclude it?)


class Actor(Module, BaseNeuralNetwork):
    """Actor neural network representing a deterministic policy used by the TD3 Agent"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_sizes: list[int],
    ):
        """
        Args:
            layer_sizes: list containing number of neurons in each hidden layer
        """
        super().__init__()

        layers = self._create_layers(
            input_dim, output_dim, layer_sizes, inc_batchnorm=True, activation=ReLU
        )

        self.network = Sequential(*layers)

    def forward(self, O: Tensor) -> Tensor:
        """Infer an action based on the observation.

        :param O: Observation tensor.
        :return: Action Tensor.
        """

        return torch.softmax(
            self.network(O), dim=1
        )  # TODO: maybe the bug is here, in the softmax dim
