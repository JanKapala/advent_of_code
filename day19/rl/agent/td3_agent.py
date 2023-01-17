# pylint: disable=invalid-name, non-ascii-name, no-member, fixme, too-many-instance-attributes, too-many-arguments, too-many-locals

"""Custom implementation of the Twin Delayed Deep Deterministic Policy Gradient algorithm"""

import itertools
import pickle
from copy import copy, deepcopy
from typing import SupportsFloat, cast

import torch
from gymnasium import Space
from gymnasium.spaces import flatten, flatten_space
from torch import Tensor
from torch.nn import DataParallel, MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from day19.rl.agent.actor import Actor
from day19.rl.agent.critic import Critic
from day19.rl.agent.replay_buffer import Record, ReplayBuffer
from day19.rl.env.action import Action
from day19.rl.env.observation import Observation

# TODO: holistic device handling
# TODO: holistic types and dtype handling
# TODO: enable some pylint checkers that has been disabled currently.

# TODO: tree search?
# TODO: noise, clamping, exploration <- there is a mess
# TODO: agent decomposition into the better classes structure


# noinspection NonAsciiCharacters
class TD3Agent:
    """Custom implementation of the Twin Delayed Deep Deterministic Policy Gradient Agent"""

    def __init__(
        self,
        action_space: Space,
        observation_space: Space,
        actor_layer_sizes: list[int],
        critic_layer_sizes: list[int],
        replay_buffer_max_size: int = int(1e4),
        batch_size: int = 128,
        γ=0.995,
        μ_θ_α=1e-4,
        Q_Φ_α=1e-3,
        ρ=0.95,
        exploration: bool = True,
        train_after: int = 128,
        learning_freq: int = 1,
        train_steps_per_update: int = 1,
        writer: SummaryWriter | None = None,
        device: str = "cpu",  # TODO: maybe do it with some pytorch class object
        act_noise: float = 0.2,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
    ) -> None:
        self.action_space = action_space
        self.observation_space = observation_space

        self.device = device

        # Actor
        self.μ_θ = Actor(
            input_dim=self._O_dim,
            output_dim=self._A_dim,
            layer_sizes=actor_layer_sizes,
        ).to(device)
        self.μ_θ_ℒ_function = (
            self.negative_mean_loss_function
        )  # Negative because gradient ascent
        self._μ_θ_α = μ_θ_α
        self.μ_θ_optimizer = Adam(self.μ_θ.parameters(), μ_θ_α)

        # Critics
        self.Q1_Φ = Critic(
            input_dim=self._O_dim + self._A_dim, layer_sizes=critic_layer_sizes
        ).to(device)
        self.Q2_Φ = Critic(
            input_dim=self._O_dim + self._A_dim, layer_sizes=critic_layer_sizes
        ).to(device)

        self.Q_Φ_ℒ_function = MSELoss()
        self._Q_Φ_α = Q_Φ_α
        self._Q_Φ_params = itertools.chain(
            self.Q1_Φ.parameters(), self.Q2_Φ.parameters()
        )
        self.Q_Φ_optimizer = Adam(self._Q_Φ_params, Q_Φ_α)

        # Target networks
        self.ρ = ρ
        self.μ_θ_targ = deepcopy(self.μ_θ)
        self.Q1_Φ_targ = deepcopy(self.Q1_Φ)
        self.Q2_Φ_targ = deepcopy(self.Q2_Φ)

        self.μ_θ_targ.eval()
        self.Q1_Φ_targ.eval()
        self.Q2_Φ_targ.eval()

        for p in self.μ_θ_targ.parameters():
            p.requires_grad = False

        for p in itertools.chain(
            self.Q1_Φ_targ.parameters(), self.Q2_Φ_targ.parameters()
        ):
            p.requires_grad = False

        # Replay Buffer
        self._batch_size = batch_size
        self._replay_buffer_max_size = replay_buffer_max_size
        self.Ɗ = ReplayBuffer(
            batch_size=batch_size,
            max_size=replay_buffer_max_size,
        )

        # Noise related stuff
        self.exploration = exploration
        self.act_noise = act_noise
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        # Other hyperparameters
        self.γ = γ
        self.policy_delay = policy_delay

        self.train_after = train_after
        self.learning_freq = learning_freq
        self.train_steps_per_update = train_steps_per_update

        # Auxiliary variables
        self._last_O: Tensor | None = None
        self._last_A: Tensor | None = None

        self.steps_counter = 0
        self.episodes_counter = 0
        self.update_times_counter = 0

        self.writer = writer

        self.returns: list[float] = []
        self._last_return: float = 0.0

    @staticmethod
    def negative_mean_loss_function(x: Tensor) -> Tensor:
        """Actor loss function"""
        return -torch.mean(x)

    def act(self, obs: Observation) -> Action:
        """Make agent to take action based on the observation.

        :param obs: Observation of the environment state.
        :return: Action that should be applied to the environment.
        """
        O = self._obs2tensor(obs)
        self._last_O = O
        with torch.no_grad():
            self.μ_θ.eval()
            A = self.μ_θ(O)

            self.Q1_Φ.eval()
            self._log_scalar("STEP/Q(A)", self.Q1_Φ(O, A))

            if self.exploration:
                A = self._add_noise_act(A)

            self.Q1_Φ.eval()
            self._log_scalar("STEP/Q(A noised)", self.Q1_Φ(O, A))

            self.μ_θ.train()

        self._last_A = A
        return self._tensor2action(A)

    @property
    def _O_dim(self) -> int:
        return cast(tuple[int], flatten_space(self.observation_space).shape)[0]

    @property
    def _A_dim(self) -> int:
        return cast(tuple[int], flatten_space(self.action_space).shape)[0]

    # noinspection PyMethodMayBeStatic
    def _tensor2action(self, A: Tensor) -> Action:
        return Action(torch.argmax(A.cpu()).item())  # TODO: device handling

    def observe(self, R: SupportsFloat, obs: Observation, terminated: bool) -> None:
        """Allow agent to observe"""
        R = float(R)
        self._log_scalar("STEP/R", R)
        self._last_return += R

        if terminated:
            return_val = copy(self._last_return)
            self.returns.append(return_val)
            self._last_return = 0.0
            self._log_at_episode_end()
            self.episodes_counter += 1

        O = self._last_O
        A = self._last_A
        R = self._reward2tensor(R)
        O_prim = self._obs2tensor(obs)
        T = self._terminated2tensor(terminated)

        self.Ɗ << (O, A, R, O_prim, T)  # pylint: disable=pointless-statement
        self.steps_counter += 1

        if self._update_time():
            self.update_times_counter += 1
            for _ in range(self.train_steps_per_update):
                self._train_step()

    def _obs2tensor(self, observation: Observation) -> Tensor:
        return (
            Tensor(flatten(self.observation_space, observation))
            .unsqueeze(0)
            .to(self.device)
        )

    # noinspection PyMethodMayBeStatic
    def _reward2tensor(self, R: float) -> Tensor:
        return Tensor([[R]])

    # noinspection PyMethodMayBeStatic
    def _terminated2tensor(self, T: bool) -> Tensor:
        return Tensor([[T]])

    # noinspection PyTypeChecker
    def _train_step(self) -> None:
        batch = next(self.Ɗ)
        batch = cast(Record, tuple(t.to(self.device) for t in batch))

        self._log_scalar("REPLAY BUFFER/R", torch.mean(batch[2]))

        self._critic_train_step(batch)
        self._actor_train_step(batch)
        self._target_nets_train_step()

    def evaluate(self, n=100) -> float:
        """Get mean return of last `n` episodes.

        :param n: The number of last episodes that should contribute to the mean return.
        :return: Mean return.
        """
        return torch.mean(torch.Tensor(self.returns[-n:])).item()

    def _critic_train_step(
        self, batch: Record
    ) -> None:  # pylint: disable=too-many-locals
        O, A, R, O_prim, T = batch

        # Critic loss calculation
        with torch.no_grad():
            self.μ_θ_targ.eval()
            A_prim = self._add_noise_observe(self.μ_θ_targ(O_prim))
            Q1_targ_action_value = self.Q1_Φ_targ(O_prim, A_prim)
            Q2_targ_action_value = self.Q2_Φ_targ(O_prim, A_prim)
            action_value = torch.min(Q1_targ_action_value, Q2_targ_action_value)

            # TODO: Better implementation, but use it latter
            # action_value = torch.min(
            #     *[
            #         Q_Φ(O_prim, A_prim).squeeze(1)
            #         for Q_Φ in [self.Q1_Φ_targ, self.Q2_Φ_targ]
            #     ]
            # )

            y = R + self.γ * (1 - T) * action_value
        self.Q1_Φ.train()
        self.Q2_Φ.train()
        y1_pred = self.Q1_Φ(O, A)
        y2_pred = self.Q2_Φ(O, A)
        Q1_Φ_ℒ = self.Q_Φ_ℒ_function(y1_pred, y)
        Q2_Φ_ℒ = self.Q_Φ_ℒ_function(y2_pred, y)
        Q_Φ_ℒ = Q1_Φ_ℒ + Q2_Φ_ℒ

        # Weights update
        self.Q_Φ_optimizer.zero_grad()
        Q_Φ_ℒ.backward()
        self.Q_Φ_optimizer.step()

    def _actor_train_step(self, batch: Record) -> None:
        # TODO: Move to common part
        if self.update_times_counter % self.policy_delay == 0:
            S, _, _, _, _ = batch

            for p in self._Q_Φ_params:  # TODO: changed, there can be a bug
                p.requires_grad = False

            # Actor loss calculation
            self.μ_θ.train()
            A = self.μ_θ(S)
            self.Q1_Φ.eval()
            action_value = self.Q1_Φ(
                S, A
            )  # TODO: maybe use: with set_grad_enabled(False)
            μ_θ_ℒ = self.μ_θ_ℒ_function(action_value)
            self._log_scalar("REPLAY BUFFER/Q(A) (mean)", torch.mean(action_value))

            # Actor weights update
            self.μ_θ_optimizer.zero_grad()
            μ_θ_ℒ.backward()
            self.μ_θ_optimizer.step()

            for p in self._Q_Φ_params:  # TODO: changed, there can be a bug
                p.requires_grad = True

    def _target_nets_train_step(self):
        # TODO: this if is probably wrong
        if self.update_times_counter % self.policy_delay == 0:
            with torch.no_grad():
                for θ, θ_targ in zip(self.μ_θ.parameters(), self.μ_θ_targ.parameters()):
                    θ_targ.data.copy_(self.ρ * θ_targ.data + (1 - self.ρ) * θ.data)

                for Q_Φ, Q_Φ_targ in [
                    (self.Q1_Φ, self.Q1_Φ_targ),
                    (self.Q2_Φ, self.Q2_Φ_targ),
                ]:
                    for Φ, Φ_targ in zip(Q_Φ.parameters(), Q_Φ_targ.parameters()):
                        Φ_targ.data.copy_(self.ρ * Φ_targ.data + (1 - self.ρ) * Φ.data)

    # DATA TRANSFORMING AND UTILITIES
    # TODO: Rethink noise adding:
    def _add_noise_act(self, A: Tensor) -> Tensor:
        noise = torch.randn_like(A).to(self.device)
        noise *= self.act_noise
        return (A + noise).clamp(min=0, max=1)

    def _add_noise_observe(self, A: Tensor) -> Tensor:
        noise = torch.randn_like(A).to(self.device)
        noise *= self.target_noise
        noise = noise.clamp(min=-self.noise_clip, max=self.noise_clip)
        return (A + noise).clamp(min=0, max=1)

    def _update_time(self):
        return (
            self.steps_counter >= self.train_after
            and self.steps_counter % self.learning_freq == 0
        )

    # LOGGING
    def _log_scalar(self, name, scalar, counter=None):
        if self.writer is None:
            return

        counter = counter or self.steps_counter
        self.writer.add_scalar(name, scalar, counter)

    def _log_nets_weights_histograms(self):
        ...
        # TODO: fix it
        # if self.writer is None:
        #     return
        #
        # nets = {
        #     "μ_θ": self.μ_θ,
        #     "Q1_Φ": self.Q1_Φ,
        #     "Q2_Φ": self.Q2_Φ,
        #     "μ_θ_targ": self.μ_θ_targ,
        #     "Q1_Φ_targ": self.Q1_Φ_targ,
        #     "Q2_Φ_targ": self.Q2_Φ_targ,
        # }
        # for net_name, net in nets.items():
        #     for param_name, param in net.named_parameters():
        #         self.writer.add_histogram(
        #             tag=f"{net_name}/{param_name}",
        #             values=param.data,
        #             global_step=self.episodes_counter,
        #         )

    def _flush_logs(self):
        if self.writer is not None:
            self.writer.flush()

    def _log_at_episode_end(self):
        self._log_nets_weights_histograms()
        self._log_scalar("Return", self.returns[-1], self.episodes_counter)
        # self._log_scalar("Replay Buffer size", len(self.Ɗ.buffer))
        self._flush_logs()

    # COPYING AND SAVING
    # TODO: casting should be probably in the other class. Maybe abstract, maybe mixin.
    def to(self, device: str) -> "TD3Agent":
        """Cast agent to the device

        :param device: "cpu" or "gpu"
        :return: Agent cast to the `device`.
        """

        writer = self.writer
        self.writer = None

        new = deepcopy(self)
        self.writer = writer
        new.writer = writer

        new.device = device

        if new.device == "cuda":
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 1:
                    new.μ_θ = DataParallel(new.μ_θ)  # type: ignore[assignment]
                    new.Q1_Φ = DataParallel(new.Q1_Φ)  # type: ignore[assignment]
                    new.Q2_Φ = DataParallel(new.Q2_Φ)  # type: ignore[assignment]
            else:
                new.device = "cpu"

        new.μ_θ_targ = deepcopy(new.μ_θ)
        new.Q1_Φ_targ = deepcopy(new.Q1_Φ)
        new.Q2_Φ_targ = deepcopy(new.Q2_Φ)
        new.μ_θ_targ.eval()
        new.Q1_Φ_targ.eval()
        new.Q2_Φ_targ.eval()

        new.μ_θ = new.μ_θ.to(new.device)
        new.Q1_Φ = new.Q1_Φ.to(new.device)
        new.Q2_Φ = new.Q2_Φ.to(new.device)
        new.μ_θ_targ = new.μ_θ_targ.to(new.device)
        new.Q1_Φ_targ = new.Q1_Φ_targ.to(new.device)
        new.Q2_Φ_targ = new.Q2_Φ_targ.to(new.device)

        # Optimizers have to be re-instantiated after moving nets to selected device
        new.μ_θ_optimizer = new.μ_θ_optimizer.__class__(new.μ_θ.parameters(), new.μ_θ_α)
        new.Q_Φ_optimizer = new.Q_Φ_optimizer.__class__(
            itertools.chain(new.Q1_Φ.parameters(), new.Q2_Φ.parameters()), new.Q_Φ_α
        )

        return new

    def save(self, file_path: str, suppress_warning: bool = False) -> None:
        """Save agent to the file.

        :param file_path: Path to the file where agent will be saved.
        :param suppress_warning: Flag, if `True` then there will be no warning printing.
        """

        writer = self.writer
        self.writer = None
        self._last_O = None  # WARNING:
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
        self.writer = writer
        if not suppress_warning:
            print(
                "Agent saved successfully! (agent.writer object can't be saved so"
                " this field has been set to `None` in saved agent, but hasn't been"
                " changed in current agent object). _last_S couldn't be saved also."
            )

    # TODO: saving and loading probably should be in the other class.
    #  maybe abstract, maybe mixin
    @classmethod
    def load(cls, file_path: str) -> "TD3Agent":
        """Load agent from the file.

        :param file_path: Path to the file where the agent has been saved.
        :return: An instance of the Agent
        """
        with open(file_path, "rb") as file:
            return pickle.load(file)

    # GETTERS AND SETTERS
    @property
    def batch_size(self):  # pylint: disable=missing-function-docstring
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size):
        self._batch_size = new_batch_size
        self.Ɗ.batch_size = new_batch_size

    @property
    def replay_buffer_max_size(self):  # pylint: disable=missing-function-docstring
        return self._replay_buffer_max_size

    @replay_buffer_max_size.setter
    def replay_buffer_max_size(self, new_value):
        self._replay_buffer_max_size = new_value
        self.Ɗ.max_size = new_value

    @property
    def μ_θ_α(self):  # pylint: disable=missing-function-docstring
        return self._μ_θ_α

    @μ_θ_α.setter
    def μ_θ_α(self, new_value):
        self._μ_θ_α = new_value
        self._set_optimizer_lr(self.μ_θ_optimizer, new_value)

    @property
    def Q_Φ_α(self):  # pylint: disable=missing-function-docstring
        return self._Q_Φ_α

    @Q_Φ_α.setter
    def Q_Φ_α(self, new_value):
        self._Q_Φ_α = new_value
        self._set_optimizer_lr(self.Q_Φ_optimizer, new_value)

    @staticmethod
    def _set_optimizer_lr(optimizer, new_lr):
        for g in optimizer.param_groups:
            g["lr"] = new_lr
