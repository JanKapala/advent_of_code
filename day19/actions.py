from enum import Enum

ORE = "ore"
CLAY = "clay"
OBSIDIAN = "obsidian"
GEODE = "geode"


class Action(Enum):
    BUILD_ORE_ROBOT = ORE
    BUILD_CLAY_ROBOT = CLAY
    BUILD_OBSIDIAN_ROBOT = OBSIDIAN
    BUILD_GEODE_ROBOT = GEODE
    BUILD_NO_ROBOT = "no_robot"

    def robot_name(self):
        if self == self.BUILD_NO_ROBOT:
            return None
        return self.value + "_robot"

    def stone_type(self):
        if self == self.BUILD_NO_ROBOT:
            return None
        return self.value

