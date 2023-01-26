"""Observation and Observation Space utilities."""

from numbers import Number
from typing import Any

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Box, Dict

from advent_of_code.day19.env.blueprint import Blueprint
from advent_of_code.day19.env.constants import (
    BLUEPRINT,
    CLAY,
    DTYPE,
    GEODE,
    HIGH,
    LOW,
    MAX_TIME,
    OBSIDIAN,
    ORE,
    ROBOTS,
    STATE,
    STONES,
    TIME,
)
from advent_of_code.day19.env.state import State


class Observation(dict):
    """Observation interface between an Agent and an Environment."""

    def __init__(self, state: State, blueprint: Blueprint, max_time: int) -> None:
        observation_dict = {
            STATE: state,
            BLUEPRINT: blueprint,
            MAX_TIME: np.array([max_time], dtype=DTYPE),
        }
        self._numpyfy_(observation_dict)
        super().__init__(observation_dict)

    # noinspection SpellCheckingInspection
    def _numpyfy_(self, obj: Any) -> None:
        """Replace observation dict values with their tensor equivalents.

        :param obj: Observation dict.
        """
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, Number):
                    obj[key] = np.array([value], dtype=DTYPE)
                else:
                    self._numpyfy_(value)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                if isinstance(value, Number):
                    obj[i] = np.array([value], dtype=DTYPE)
                else:
                    self._numpyfy_(value)


def generate_observation_space(max_time: int, robots_costs_boundaries) -> Space:
    """Generate an observation space compatible with the `Observation` class.

    :param max_time: Max time of the game.
    :param robots_costs_boundaries: min and max costs generally.
    :return:
    """
    max_stones = (1 + max_time) * max_time / 2
    max_robots = max_time + 1
    return Dict(
        {
            STATE: Dict(
                {
                    ROBOTS: Dict(
                        {
                            ORE: Box(1, max_robots, dtype=DTYPE),
                            CLAY: Box(0, max_robots, dtype=DTYPE),
                            OBSIDIAN: Box(0, max_robots, dtype=DTYPE),
                            GEODE: Box(0, max_robots, dtype=DTYPE),
                        }
                    ),
                    STONES: Dict(
                        {
                            ORE: Box(0, max_stones, dtype=DTYPE),
                            CLAY: Box(0, max_stones, dtype=DTYPE),
                            OBSIDIAN: Box(0, max_stones, dtype=DTYPE),
                            GEODE: Box(0, max_stones, dtype=DTYPE),
                        }
                    ),
                    TIME: Box(0, max_time, dtype=DTYPE),
                }
            ),
            BLUEPRINT: Dict(
                {
                    ORE: Dict(
                        {
                            ORE: Box(
                                robots_costs_boundaries[ORE][ORE][LOW],
                                robots_costs_boundaries[ORE][ORE][HIGH],
                                dtype=DTYPE,
                            )
                        }
                    ),
                    CLAY: Dict(
                        {
                            ORE: Box(
                                robots_costs_boundaries[CLAY][ORE][LOW],
                                robots_costs_boundaries[CLAY][ORE][HIGH],
                                dtype=DTYPE,
                            )
                        }
                    ),
                    OBSIDIAN: Dict(
                        {
                            ORE: Box(
                                robots_costs_boundaries[OBSIDIAN][ORE][LOW],
                                robots_costs_boundaries[OBSIDIAN][ORE][HIGH],
                                dtype=DTYPE,
                            ),
                            CLAY: Box(
                                robots_costs_boundaries[OBSIDIAN][CLAY][LOW],
                                robots_costs_boundaries[OBSIDIAN][CLAY][HIGH],
                                dtype=DTYPE,
                            ),
                        }
                    ),
                    GEODE: Dict(
                        {
                            ORE: Box(
                                robots_costs_boundaries[GEODE][ORE][LOW],
                                robots_costs_boundaries[GEODE][ORE][HIGH],
                                dtype=DTYPE,
                            ),
                            OBSIDIAN: Box(
                                robots_costs_boundaries[GEODE][OBSIDIAN][LOW],
                                robots_costs_boundaries[GEODE][OBSIDIAN][HIGH],
                                dtype=DTYPE,
                            ),
                        }
                    ),
                }
            ),
            MAX_TIME: Box(0, 64, dtype=DTYPE),
        }
    )
