# pylint: disable=fixme

"""Blueprint of the robots"""

from advent_of_code.day19.env.constants import CLAY, GEODE, OBSIDIAN, ORE

# TODO: check if this pretty dict can be used in the Blueprint class.
# from day19.rl.env.utils import PrettyDict


class Blueprint(dict):
    """Blueprint of the robots"""

    def __init__(  # pylint: disable=too-many-arguments
        self, ore_ore, clay_ore, obsidian_ore, obsidian_clay, geode_ore, geode_obsidian
    ) -> None:
        super().__init__(
            {
                ORE: {
                    ORE: ore_ore,
                },
                CLAY: {
                    ORE: clay_ore,
                },
                OBSIDIAN: {
                    ORE: obsidian_ore,
                    CLAY: obsidian_clay,
                },
                GEODE: {ORE: geode_ore, OBSIDIAN: geode_obsidian},
            }
        )
