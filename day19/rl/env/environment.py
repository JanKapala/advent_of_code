import json
from copy import deepcopy
from math import inf
from typing import Any, SupportsFloat

from gymnasium import Env
from gymnasium.core import RenderFrame
from gymnasium.envs.registration import EnvSpec
from gymnasium.spaces import flatten, flatten_space

from day19.rl.data_loading import load_blueprints, extract_global_data, INPUT_FILE_PATH
from day19.rl.env.action import ACTION_TO_ROBOT_TYPE_MAPPING, _get_actions_space
from day19.rl.env.blueprints import Blueprint
from day19.rl.env.constants import ROBOTS, STONES, STONE_TYPES, TIME, GEODE
from day19.rl.env.observation import Observation, generate_observation_space
from day19.rl.env.state import State


class NotEnoughMineralsEnv(Env):
    metadata: dict[str, Any] = {
        "render_modes": [None, "human", "rgb_array", "ansi", "rgb_array_list", "ansi_list"]
        # TODO: research what is it all about
    }

    def __init__(self,
                 blueprint: Blueprint,
                 max_time: int,
                 robots_costs_boundaries: dict[str, dict[str, dict[str, int]]],
                 render_mode: str | None = None
                 ):
        self.blueprint = blueprint
        self.max_time = max_time
        self.robots_costs_boundaries = robots_costs_boundaries
        self._last_state = None

        self.observation_space = generate_observation_space(self.max_time, self.robots_costs_boundaries)
        self.action_space = _get_actions_space()

        # TODO: Objects related to render_modes should eb initialized here.
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.reward_range = (-inf, inf)  # TODO: change to desired
        self.spec: EnvSpec | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[
        Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._last_state = State()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        old_robots = deepcopy(self._last_state[ROBOTS])

        # Produce robot if needed and possible
        robot_type = ACTION_TO_ROBOT_TYPE_MAPPING[action]
        # Assumption: when agent tries to make action that is not available then the Empty/None/Null action is taken.
        # It sounds reasonable because it's like trying to do something impossible - if you try to do it, nothing happens.
        if robot_type is not None:  # TODO: if there is a bug, it can be related to this line!!!
            new_stones = {st: self._last_state[STONES][st] - self.blueprint[robot_type].get(st, 0) for st in
                          STONE_TYPES}
            if min(new_stones.values()) >= 0:
                self._last_state[STONES] = new_stones
                self._last_state[ROBOTS][robot_type] += 1

        # Produce stones based on the original robots number
        self._last_state[STONES] = {st: self._last_state[STONES][st] + old_robots[st] for st in STONE_TYPES}

        # Update time
        self._last_state[TIME] += 1

        # Produce result
        observation = self._get_obs()
        reward = self._last_state[STONES][GEODE]
        terminated = self._last_state[TIME] == self.max_time
        truncated = False  # TODO: maybe there is an use case for it?
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> Observation:
        return Observation(self._last_state, self.blueprint, self.max_time)

    def _get_info(self) -> dict[str, Any]:
        return {}  # TODO: maybe there is an use case for it?

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        ...  # TODO

    def close(self) -> None:
        ...  # TODO


if __name__ == "__main__":
    blueprints = load_blueprints(INPUT_FILE_PATH)
    robots_costs_boundaries, costs_boundaries = extract_global_data(blueprints)

    max_time = 24

    env = NotEnoughMineralsEnv(
        blueprint=blueprints[0],
        max_time=max_time,
        robots_costs_boundaries=robots_costs_boundaries
    )

    obs, _ = env.reset()

    print(f"observation: {obs}")
    print(f"observation type: {type(obs)}")
    flatten_observation = flatten(env.observation_space, obs)
    print(f"flatten observation: {flatten_observation}")
    print(f"flatten observation type: {type(flatten_observation)}")

    action = env.action_space.sample()
    print(f"action: {action}")
    print(f"action type: {type(action)}")
    flatten_action = flatten(env.action_space, action)
    print(f"flatten action: {flatten_action}")
    print(f"flatten action type: {type(flatten_action)}")

    print(obs)

    obs, _, _, _, _ = env.step(1)

    print(obs)

    fs = flatten_space(env.observation_space)


    print("xd")
    print("lol")