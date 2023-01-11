from copy import deepcopy
from typing import Union

import pandas as pd

from actions import Action
from blueprint import Blueprint


class StonesLike:
    def __init__(self, ore: int = 0, clay: int = 0, obsidian: int = 0, geode: int = 0) -> None:
        self.ore = ore
        self.clay = clay
        self.obsidian = obsidian
        self.geode = geode

    def __eq__(self, other) -> bool:
        return (
            self.ore == other.ore
            and self.clay == other.clay
            and self.obsidian == other.obsidian
            and self.geode == other.geode
        )


class State:
    def __init__(
        self,
        blueprint: Blueprint,
        resources: StonesLike = None,
        robots: StonesLike = None,
        time: int = 0,
    ) -> None:
        self.blueprint = blueprint
        self.resources = resources or StonesLike()
        self.robots = robots or StonesLike(ore=1)
        self.time = time
        self.previous = None

    def __gt__(self, other: "State") -> bool:  # TODO: test it
        stone_types = ["ore", "clay", "obsidian", "geode"]

        # Rule I
        # Every state is better than state with more robots of some type than max cost of this type.
        stone_types_without_geode = deepcopy(stone_types)
        stone_types_without_geode.remove("geode")
        if any([getattr(other.robots, stone_type) > getattr(other.blueprint.max_costs, stone_type) for stone_type in stone_types_without_geode]):
            return True

        # Rule II
        # Strictly better state wins
        if all(
            [
                getattr(self.resources, resource_type)
                >= getattr(other.resources, resource_type)
                and getattr(self.robots, resource_type)
                >= getattr(other.robots, resource_type)
                for resource_type in stone_types
            ]
            + [self.time <= other.time]
        ):
            return True



        return False

    def __lt__(self, other: "State") -> bool:  # TODO: test it
        return other > self

    def __eq__(self, other: "State") -> bool:
        for resource_type in ["ore", "clay", "obsidian"]:
            if (
                getattr(self.robots, resource_type)
                < getattr(self.blueprint.max_costs, resource_type)
                or getattr(other.robots, resource_type)
                < getattr(other.blueprint.max_costs, resource_type)
            ) and getattr(self.resources, resource_type) != getattr(
                other.resources, resource_type
            ):
                return False
        if self.resources.geode != other.resources.geode or self.time != other.time:
            return False
        return True

    def build(self, action: Action) -> Union["State", None]:
        if any([getattr(self.resources, stone_type) - cost < 0 for stone_type, cost in getattr(self.blueprint, action.robot_name()).__dict__.items()]):
            return self

        for stone_type, cost in getattr(self.blueprint, action.robot_name()).__dict__.items():
            setattr(self.resources, stone_type, getattr(self.resources, stone_type)-cost)
        setattr(self.robots, action.stone_type(), getattr(self.robots, action.stone_type())+1)
        return self

    def next(self, action: Action) -> Union["State", None]:
        next_state = deepcopy(self)  # TODO: verify validity of the transition
        if action.value != Action.BUILD_NO_ROBOT.value:
            next_state = next_state.build(action)

        next_state.time += 1
        for resource_type in ["ore", "clay", "obsidian", "geode"]:
            setattr(
                next_state.resources,
                resource_type,
                getattr(next_state.resources, resource_type)
                + getattr(self.robots, resource_type),
            )
        next_state.previous = self

        return next_state

    def to_dataframe(self) -> pd.DataFrame:
        resources_types = ["ore", "clay", "obsidian", "geode"]
        columns = ["resources", "robots"]

        leveled_columns = [(f"{self.time}m", "S"), (f"{self.time}m", "R")]

        df = pd.DataFrame(
            [
                [getattr(getattr(self, column), rt) for column in columns]
                for rt in resources_types
            ]
        )
        df.index = resources_types
        df.columns = columns
        df.columns = pd.MultiIndex.from_tuples(leveled_columns)

        return df

    def __str__(self) -> str:
        return self.to_dataframe().__str__()

    def __repr__(self) -> str:
        return self.to_dataframe().__repr__()

    @property
    def history(self):
        states = []
        temp_state = self
        states.append(temp_state.to_dataframe())
        while temp_state.previous is not None:
            temp_state = temp_state.previous
            states.append(temp_state.to_dataframe())
        return states

    @property
    def history_dataframe(self):
        return pd.concat(reversed(self.history), axis=1)

