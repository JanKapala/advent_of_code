# pylint: disable=fixme

"""Common constants"""

# TODO: it would be ideal if this module doesn't exist.
#  try to move each constant to its best location (ideally near the origin of usage)
import os

import numpy as np

from advent_of_code.day19.constants import DAY_19_ROOT_DIR

STONES = "stones"
ROBOTS = "robots"
ORE = "ore"
CLAY = "clay"
OBSIDIAN = "obsidian"
GEODE = "geode"
TIME = "time"
ROBOT_TYPES = (ORE, CLAY, OBSIDIAN, GEODE)
STONE_TYPES = ROBOT_TYPES
LOW = "LOW"
HIGH = "HIGH"
STATE = "state"
BLUEPRINT = "blueprint"
MAX_TIME = "max_time"
DTYPE = np.int64
ASSETS_PATH = os.path.join(DAY_19_ROOT_DIR, "env/rendering/assets")
