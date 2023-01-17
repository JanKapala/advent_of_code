# pylint: disable=invalid-name, too-many-arguments, no-member, no-name-in-module

"""Critic Model implementation"""

from torch import Tensor, cat
from torch.nn import Module, ReLU, Sequential

from day19.rl.agent.base_neural_network import BaseNeuralNetwork


class Critic(Module, BaseNeuralNetwork):
    """Critic neural network representing used by the TD3 Agent"""

    def __init__(
        self,
        input_dim: int,
        layer_sizes: list[int],
    ):
        super().__init__()
        layers = self._create_layers(
            input_dim, 1, layer_sizes, inc_batchnorm=True, activation=ReLU
        )
        self.network = Sequential(*layers)

    def forward(self, O: Tensor, A: Tensor) -> Tensor:
        """Performs forward propagation."""

        return self.network(cat([O, A], dim=1))
