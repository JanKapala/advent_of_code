# TODO


from torch import rand, Tensor

from day19.rl.agent.critic import Critic


def test_critic():
    O_shape = 16
    A_shape = 5
    shape = O_shape + A_shape

    batch_size = 16

    O = rand((batch_size, O_shape))
    A = rand(batch_size, A_shape)
    critic = Critic(input_dim=shape, layer_sizes=[4, 8, 4])

    Q_A = critic(O, A)

    assert Q_A is Tensor
