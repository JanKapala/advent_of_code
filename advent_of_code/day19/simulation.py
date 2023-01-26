# pylint: disable=fixme, too-many-arguments

"""Simulation of the dynamics between an Agent and an Environment that enables experimentation."""

from gymnasium import Env
from tqdm.auto import tqdm, trange

from advent_of_code.day19.agent.td3_agent import TD3Agent

# TODO:
#  Rethink simulation, agent and environment relations - create new architecture that:
#  -> takes into account agent self reflection
#  -> allows for multiple and dynamically spawned agents
#  -> deals with applying of multiple actions in the same timestep.
#  Reimplement simulation

# TODO: hyperparameters tuning: Population based training
# TODO: parallelization, local/cluster deployments


def simulate(
    env: Env,
    agent: TD3Agent,  # TODO: create an abstract agent class
    episodes: int,
    max_episode_steps: int,
    render: bool = True,
    seed: int | None = None,
    episodes_pb=True,  # TODO: maybe remove this argument.
    steps_pb=True,  # TODO: maybe remove this argument.
) -> None:  # TODO: Maybe simulation should return some results that could be analysed/saved.
    """Simulate dynamics between an Agent and an Environment.

    :param env: Environment.
    :param agent: Reinforcement Learning Agent.
    :param episodes: Number of episodes to play.
    :param max_episode_steps: Max number of steps in the episode.
    :param render: Flag, if true then episodes will be rendered.
    :param seed: PRNG seed.
    :param episodes_pb:
    :param steps_pb:
    :return:
    """

    for _ in trange(episodes, position=0, leave=True, disable=(not episodes_pb)):
        obs, _ = env.reset(seed=seed)
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
