from numbers import Number

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Dict, Box

from day19.rl.env.constants import STATE, BLUEPRINT, MAX_TIME, DTYPE, ROBOTS, ORE, CLAY, OBSIDIAN, GEODE, STONES, TIME, \
    LOW, HIGH


class Observation(dict):
    def __init__(self, state, blueprint, max_time):
        d = {
            STATE: state,
            BLUEPRINT: blueprint,
            MAX_TIME: np.array([max_time], dtype=DTYPE)
        }
        self._numpyfy(d)
        super().__init__(d)

    def _numpyfy(self, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, Number):
                    obj[k] = np.array([v], dtype=DTYPE)
                else:
                    self._numpyfy(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, Number):
                    obj[i] = np.array([v], dtype=DTYPE)
                else:
                    self._numpyfy(v)


def generate_observation_space(max_time, robots_costs_boundaries) -> Space:
    max_stones = (1 + max_time) * max_time / 2
    max_robots = max_time + 1
    return Dict({
        STATE: Dict({
            ROBOTS: Dict({
                ORE: Box(1, max_robots, dtype=DTYPE),
                CLAY: Box(0, max_robots, dtype=DTYPE),
                OBSIDIAN: Box(0, max_robots, dtype=DTYPE),
                GEODE: Box(0, max_robots, dtype=DTYPE),
            }),
            STONES: Dict({
                ORE: Box(0, max_stones, dtype=DTYPE),
                CLAY: Box(0, max_stones, dtype=DTYPE),
                OBSIDIAN: Box(0, max_stones, dtype=DTYPE),
                GEODE: Box(0, max_stones, dtype=DTYPE),
            }),
            TIME: Box(0, max_time, dtype=DTYPE),
        }),
        BLUEPRINT: Dict({
            ORE: Dict({
                ORE: Box(robots_costs_boundaries[ORE][ORE][LOW],
                         robots_costs_boundaries[ORE][ORE][HIGH],
                         dtype=DTYPE)
            }),
            CLAY: Dict({
                ORE: Box(robots_costs_boundaries[CLAY][ORE][LOW],
                         robots_costs_boundaries[CLAY][ORE][HIGH],
                         dtype=DTYPE)
            }),
            OBSIDIAN: Dict({
                ORE: Box(robots_costs_boundaries[OBSIDIAN][ORE][LOW],
                         robots_costs_boundaries[OBSIDIAN][ORE][HIGH],
                         dtype=DTYPE),
                CLAY: Box(robots_costs_boundaries[OBSIDIAN][CLAY][LOW],
                          robots_costs_boundaries[OBSIDIAN][CLAY][HIGH],
                          dtype=DTYPE)
            }),
            GEODE: Dict({
                ORE: Box(robots_costs_boundaries[GEODE][ORE][LOW],
                         robots_costs_boundaries[GEODE][ORE][HIGH],
                         dtype=DTYPE),
                OBSIDIAN: Box(robots_costs_boundaries[GEODE][OBSIDIAN][LOW],
                              robots_costs_boundaries[GEODE][OBSIDIAN][HIGH],
                              dtype=DTYPE)
            }),
        }),
        MAX_TIME: Box(0, 64, dtype=DTYPE)
    })
