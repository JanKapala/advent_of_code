import os
import random
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from advent_of_code.day19.agent.td3_agent import TD3Agent
from advent_of_code.day19.data_loading import extract_global_data, load_blueprints
from advent_of_code.day19.env.environment import NotEnoughMineralsEnv
from advent_of_code.day19.simulation import simulate
from constants import DAY_19_INPUT_FILE_PATH, DAY_19_LOG_DIR

# TODO: repo related stuff
# TODO: scale reward and return, or maybe
#  better: create interpreter (related to the architecture change)


if __name__ == "__main__":
    # General
    SEED = 0
    torch.manual_seed(SEED)
    random.seed(SEED)

    writer = SummaryWriter(
        log_dir=os.path.join(
            DAY_19_LOG_DIR, datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        )
    )

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
    layers = [32, 16, 8]
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
        seed=SEED,
    )
