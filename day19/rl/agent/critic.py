# pylint: disable=invalid-name, too-many-arguments, no-member, no-name-in-module

"""Critic Model implementation"""

from torch import Tensor, cat, rand
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


if __name__ == "__main__":
    O_shape = 16
    A_shape = 5
    shape = O_shape + A_shape

    batch_size = 16

    O = rand((batch_size, O_shape))
    A = rand(batch_size, A_shape)
    critic = Critic(input_dim=shape, layer_sizes=[4, 8, 4])

    A = critic(O, A)

    for p in critic.parameters():
        print(p.data)
