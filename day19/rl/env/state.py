"""State of the environment NotEnoughMineralsEnv"""

from day19.rl.env.constants import CLAY, GEODE, OBSIDIAN, ORE, ROBOTS, STONES, TIME


class State(dict):
    """State of the environment NotEnoughMineralsEnv"""

    def __init__(self):
        super().__init__(
            {
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
            }
        )