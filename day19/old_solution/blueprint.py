from abc import ABC
from dataclasses import dataclass


@dataclass
class Robot(ABC):
    ore: int


@dataclass
class OreRobot(Robot):
    pass


@dataclass
class ClayRobot(Robot):
    pass


@dataclass
class ObsidianRobot(Robot):
    clay: int


@dataclass
class GeodeRobot(Robot):
    obsidian: int


@dataclass
class MaxCosts:
    ore: int
    clay: int
    obsidian: int


@dataclass
class Blueprint:
    ore_robot: OreRobot
    clay_robot: ClayRobot
    obsidian_robot: ObsidianRobot
    geode_robot: GeodeRobot
    n: int

    @property
    def max_costs(self) -> MaxCosts:
        return MaxCosts(
            ore=max(
                [
                    self.ore_robot.ore,
                    self.clay_robot.ore,
                    self.obsidian_robot.ore,
                    self.geode_robot.ore,
                ]
            ),
            clay=self.obsidian_robot.clay,
            obsidian=self.geode_robot.obsidian,
        )


# TODO: tests
# from memory import Memory
# from state import StonesLike
# m = Memory()
# max_costs = MaxCosts(4, 20, 20)
# s1 = State(max_costs=max_costs)
# s2 = State(max_costs=max_costs)
# m.put(s1)
# m.put(s2)
# assert len(m.queue) == 1
#
# s1, s2 = State(max_costs), State(max_costs)
# assert s1 == s2
#
# r1, r2 = StonesLike(), StonesLike()
# assert r1 == r2
#
# l = [s1]
# assert s2 in l
#
# l = [r1]
# assert r2 in l