# pylint: disable=fixme, too-many-arguments, non-ascii-name

"""Simulation of the dynamics between an Agent and an Environment that enables experimentation."""

import os
import random
import string
from datetime import datetime
from time import sleep

import torch
from gymnasium import Env
from gymnasium.utils.env_checker import check_env
from gymnasium.utils.play import play
from pygame import K_SPACE
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange

from day19.constants import DAY_19_INPUT_FILE_PATH, DAY_19_LOG_DIR
from day19.rl.agent.td3_agent import TD3Agent
from day19.rl.data_loading import extract_global_data, load_blueprints
from day19.rl.env.action import Action
from day19.rl.env.environment import NotEnoughMineralsEnv, Noop


# TODO:
#  Rethink simulation, agent and environment relations - create new architecture that:
#  -> takes into account agent self reflection
#  -> allows for multiple and dynamically spawned agents
#  -> deals with applying of multiple actions in the same timestep.
#  Reimplement simulation

# TODO: hyperparameters tunning
# TODO: Population based training
# TODO: parallelization, local/cluster deployments


def simulate(
    env: Env,
    agent: TD3Agent,  # TODO: create an abstract agent class
    episodes: int,
    max_episode_steps: int,
    render: bool = True,
    episodes_pb=True,  # TODO: maybe remove this argument.
    steps_pb=True,  # TODO: maybe remove this argument.
) -> None:  # TODO: Maybe simulation should return some results that could be analysed/saved.
    """Simulate dynamics between an Agent and an Environment.

    :param env: Environment.
    :param agent: Reinforcement Learning Agent.
    :param episodes: Number of episodes to play.
    :param max_episode_steps: Max number of steps in the episode.
    :param render: Flag, if true then episodes will be rendered.
    :param episodes_pb: TODO
    :param steps_pb: TODO
    :return:
    """

    for _ in trange(episodes, position=0, leave=True, disable=(not episodes_pb)):
        obs, _ = env.reset()
        episode_steps = 0

        with tqdm(
            total=max_episode_steps, position=1, leave=False, disable=(not steps_pb)
        ) as pbar:
            while True:
                if render:
                    env.render()

                action = agent.act(obs)
                next_obs, reward, terminated, _, _ = env.step(action)

                if episode_steps >= max_episode_steps:
                    terminated = True
                agent.observe(reward, next_obs, terminated)

                obs = next_obs
                episode_steps += 1
                pbar.update(1)

                if terminated:
                    break
    env.close()


def generate_random_string(length):  # pylint: disable=missing-function-docstring
    return "".join(random.choices(string.ascii_lowercase, k=length))


if __name__ == "__main__":
    # General
    SEED = 0
    torch.manual_seed(SEED)
    random.seed(SEED)

    real_logdir = os.path.join(
        DAY_19_LOG_DIR, datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    )
    writer = SummaryWriter(log_dir=real_logdir)

    # Environment
    blueprints = load_blueprints(DAY_19_INPUT_FILE_PATH)
    robots_costs_boundaries, costs_boundaries = extract_global_data(blueprints)

    MAX_TIME = 24
    environment = NotEnoughMineralsEnv(
        blueprint=blueprints[0],
        max_time=MAX_TIME,
        robots_costs_boundaries=robots_costs_boundaries,
    )

    # Agent
    layers = [14, 10, 6]
    training_agent = TD3Agent(
        action_space=environment.action_space,
        observation_space=environment.observation_space,
        actor_layer_sizes=layers,  # (64, 128, 64),
        critic_layer_sizes=layers,  # (64, 128, 64),
        replay_buffer_max_size=int(10e4),
        batch_size=64,
        γ=1,
        μ_θ_α=1e-5,
        Q_Φ_α=1e-3,
        ρ=0.95,
        exploration=True,
        train_after=64,
        learning_freq=1,
        train_steps_per_update=1,
        writer=writer,
        device="cpu",
        act_noise=0.4,
        target_noise=0.1,
        noise_clip=0.5,
        policy_delay=2,
    )

    # print(check_env(environment))



    # Training
    simulate(
        environment,
        training_agent,
        episodes=1600,
        render=False,
        max_episode_steps=MAX_TIME,
    )

    # Evaluation
    training_agent.exploration = False
    simulate(
        environment,
        training_agent,
        episodes=100,
        render=False,
        max_episode_steps=MAX_TIME,
    )


# TODO:
#  -> prioritized replay buffer
#  -> noise, clamping, exploration <- there is a mess
#  -> repo related stuff
#  -> publish env
