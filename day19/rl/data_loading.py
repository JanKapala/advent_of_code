from math import inf

from day19.rl.env.blueprints import Blueprint
from day19.rl.env.utils import PrettyDict
from day19.rl.env.constants import ORE, CLAY, OBSIDIAN, GEODE, ROBOT_TYPES, LOW, HIGH

INPUT_FILE_PATH = "./../../input.txt"  # TODO: do it better


def load_blueprints(filepath: str) -> list[Blueprint]:
    blueprints = []
    with open(filepath) as file:
        lines = file.readlines()

        for line in lines:
            match line.split():
                case [
                    "Blueprint",
                    _n,
                    "Each",
                    "ore",
                    "robot",
                    "costs",
                    ore_ore,
                    "ore.",
                    "Each",
                    "clay",
                    "robot",
                    "costs",
                    clay_ore,
                    "ore.",
                    "Each",
                    "obsidian",
                    "robot",
                    "costs",
                    obsidian_ore,
                    "ore",
                    "and",
                    obsidian_clay,
                    "clay.",
                    "Each",
                    "geode",
                    "robot",
                    "costs",
                    geode_ore,
                    "ore",
                    "and",
                    geode_obsidian,
                    "obsidian.",
                ]:
                    blueprints.append(
                        Blueprint(
                            int(ore_ore),
                            int(clay_ore),
                            int(obsidian_ore),
                            int(obsidian_clay),
                            int(geode_ore),
                            int(geode_obsidian)
                        )
                    )

                case _:
                    raise IOError("Invalid blueprint input")
    return blueprints


def extract_global_data(blueprints: list[Blueprint]):
    robots_costs_boundaries = PrettyDict({
        ORE: {
            ORE: {LOW: inf, HIGH: -inf}
        },
        CLAY: {
            ORE: {LOW: inf, HIGH: -inf}
        },
        OBSIDIAN: {
            ORE: {LOW: inf, HIGH: -inf},
            CLAY: {LOW: inf, HIGH: -inf}
        },
        GEODE: {
            ORE: {LOW: inf, HIGH: -inf},
            OBSIDIAN: {LOW: inf, HIGH: -inf}
        },
    })

    costs_boundaries = PrettyDict({
        ORE: {LOW: inf, HIGH: -inf},
        CLAY: {LOW: inf, HIGH: -inf},
        OBSIDIAN: {LOW: inf, HIGH: -inf},
    })

    for blueprint in blueprints:
        for robot, stone in [(ORE, ORE), (CLAY, ORE), (OBSIDIAN, ORE), (OBSIDIAN, CLAY), (GEODE, ORE),
                             (GEODE, OBSIDIAN)]:
            robots_costs_boundaries[robot][stone][LOW] = min(blueprint[robot][stone],
                                                             robots_costs_boundaries[robot][stone][LOW])
            robots_costs_boundaries[robot][stone][HIGH] = max(blueprint[robot][stone],
                                                              robots_costs_boundaries[robot][stone][HIGH])

    for robot in ROBOT_TYPES:
        for stone in (ORE, CLAY, OBSIDIAN):
            costs_boundaries[stone][LOW] = min(robots_costs_boundaries[robot].get(stone, {}).get(LOW, inf),
                                               costs_boundaries[stone][LOW])
            costs_boundaries[stone][HIGH] = max(robots_costs_boundaries[robot].get(stone, {}).get(HIGH, -inf),
                                                costs_boundaries[stone][HIGH])

    return robots_costs_boundaries, costs_boundaries
