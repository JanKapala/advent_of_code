from day19.rl.env.constants import ORE, CLAY, OBSIDIAN, GEODE, STONE_TYPES
from day19.rl.env.utils import PrettyDict


class Blueprint(dict):
    def __init__(self, ore_ore, clay_ore, obsidian_ore, obsidian_clay, geode_ore, geode_obsidian):
        super().__init__({
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
            GEODE: {
                ORE: geode_ore,
                OBSIDIAN: geode_obsidian
            }
        })
