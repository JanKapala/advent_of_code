# pylint: disable=fixme, too-many-instance-attributes

"""Not Enough Material Environment from the Day 19 of the Advent of Code 2022."""

from copy import deepcopy
from math import inf
from typing import Any, SupportsFloat

from gymnasium import Env
from gymnasium.core import RenderFrame
from gymnasium.envs.registration import EnvSpec

from day19.constants import DAY_19_INPUT_FILE_PATH
from day19.rl.data_loading import load_blueprints, extract_global_data
from day19.rl.env.action import (
    ACTION_TO_ROBOT_TYPE_MAPPING,
    Action,
    generate_action_space,
)
from day19.rl.env.blueprints import Blueprint
from day19.rl.env.constants import GEODE, ROBOTS, STONE_TYPES, STONES, TIME
from day19.rl.env.observation import Observation, generate_observation_space
from day19.rl.env.rendering.renderer import Renderer
from day19.rl.env.state import State

# TODO: play utility from gymnasium
# TODO: logging with structlog and tensorboard
# TODO: pygame


class Noop(Action):
    pass


class NotEnoughMineralsEnv(Env):
    """ "Not Enough Materials" environment.
    It represents the problem described in the Day 19 of the Advent of Code 2022 challenge.
    Full description of the problem and input data can be found here:
     https://adventofcode.com/2022/day/19
    Use functions from `data_loading` module to obtain data needed for the creation of the
     environment.
    """

    metadata: dict[str, Any] = {
        "render_modes": [
            None,
            "human",
            "rgb_array",
            # "ansi",
            # "rgb_array_list",
            # "ansi_list",
        ]
        # TODO: research what is it all about
    }

    # noinspection PyShadowingNames
    def __init__(
        self,
        blueprint: Blueprint,
        max_time: int,
        robots_costs_boundaries: dict[str, dict[str, dict[str, int]]],
        render_mode: str | None = None,
    ):
        self.blueprint = blueprint
        self.max_time = max_time
        self.robots_costs_boundaries = robots_costs_boundaries
        self._last_state: State = State()

        self.observation_space = generate_observation_space(
            self.max_time, self.robots_costs_boundaries
        )
        self.action_space = generate_action_space()

        # TODO: Objects related to render_modes should eb initialized here.
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode)
        self.reward_range = (0, inf)
        self.spec: EnvSpec | None = None
        self._keys_to_action = {
            key: Action(i) for i, key in enumerate(["a", "s", "d", "f", " "])
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        self._last_state = State()
        return self._get_obs(), self._get_info()

    # noinspection PyShadowingNames
    def step(
        self, action: Action | Noop
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        if action != Action.NOOP:  # TODO: it's hack for making it playable, do it better
            old_robots = deepcopy(self._last_state[ROBOTS])

            # Produce robot if needed and possible
            robot_type = ACTION_TO_ROBOT_TYPE_MAPPING[action]
            # Assumption: when agent tries to make action that is not available
            # then the Empty/None/Null action is taken. It sounds reasonable
            # because it's like trying to do something impossible - if you try
            # to do it, nothing happens.

            # TODO: if there is a bug, it can be related to this line!!!
            if robot_type is not None:
                new_stones = {
                    st: self._last_state[STONES][st] - self.blueprint[robot_type].get(st, 0)
                    for st in STONE_TYPES
                }
                if min(new_stones.values()) >= 0:
                    self._last_state[STONES] = new_stones
                    self._last_state[ROBOTS][robot_type] += 1

            # Produce stones based on the original robots number
            self._last_state[STONES] = {
                st: self._last_state[STONES][st] + old_robots[st] for st in STONE_TYPES
            }

            # Update time
            self._last_state[TIME] += 1

        # Produce result
        observation = self._get_obs()
        reward = float(self._last_state[STONES][GEODE])
        terminated = self._last_state[TIME].item() == self.max_time
        truncated = False  # TODO: maybe there is an use case for it?
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> Observation:
        return Observation(self._last_state, self.blueprint, self.max_time)

    # noinspection PyMethodMayBeStatic
    def _get_info(self) -> dict[str, Any]:
        return {}  # TODO: maybe there is an use case for it?

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.renderer.render(self._get_obs())

    def close(self) -> None:
        self.renderer.close()

    def get_keys_to_action(self):
        return self._keys_to_action


if __name__ == "__main__":
    blueprints = load_blueprints(DAY_19_INPUT_FILE_PATH)
    robots_costs_boundaries, costs_boundaries = extract_global_data(blueprints)

    max_time = 24

    env = NotEnoughMineralsEnv(
        blueprint=blueprints[0],
        max_time=max_time,
        robots_costs_boundaries=robots_costs_boundaries,
    )

    obs, _ = env.reset()
    env.render()
