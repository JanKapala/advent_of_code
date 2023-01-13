
from day19.rl.env.constants import STONES, ROBOTS, ORE, CLAY, OBSIDIAN, GEODE, TIME


class State(dict):
    def __init__(self):
        super().__init__({
            ROBOTS: {
                ORE: 1,
                CLAY: 0,
                OBSIDIAN: 0,
                GEODE: 0,
            },
            STONES: {
                ORE: 0,
                CLAY: 0,
                OBSIDIAN: 0,
                GEODE: 0,
            },
            TIME: 0,
        })





