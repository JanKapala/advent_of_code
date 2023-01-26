"""Indicators needed for the rendering of the environment scene."""

from abc import abstractmethod, ABC

import pygame

from advent_of_code.day19.env.constants import (
    STATE,
    STONE_TYPES,
    STONES,
    ROBOTS,
    BLUEPRINT,
    ORE,
    ROBOT_TYPES,
    OBSIDIAN,
    CLAY,
    GEODE,
    TIME,
    MAX_TIME,
)
from advent_of_code.day19.env.observation import Observation
from advent_of_code.day19.env.rendering.colors import BLACK
from advent_of_code.day19.env.rendering.scene_components.scene_component import (
    SceneComponent,
)


class Indicators(SceneComponent):
    """Base class for indicators"""

    def __init__(self, canvas, window_size: int, box_size: int) -> None:
        """

        :param canvas: Pygame Canvas.
        :param window_size:
        :param box_size:
        """
        super().__init__(canvas, window_size, box_size)

        self.horizontal_resources_text_offset: int = int(0.3 * self.box_size)
        self.vertical_resources_text_offset: int = int(0.35 * self.box_size)

        self.font = pygame.font.SysFont("chalkduster.ttf", int(self.box_size * 0.6))

    @abstractmethod
    def render(self, obs: Observation) -> None:
        """Render into the canvas.

        :param obs: An element of the environment’s observation_space.
        :return:
        """

        raise NotImplementedError()


class ResourcesIndicators(Indicators, ABC):
    resource_type: str = None

    def __init__(self, canvas, window_size: int, box_size: int) -> None:
        super().__init__(canvas, window_size, box_size)

    def render(self, obs: Observation) -> None:
        resources = [str(obs[STATE][self.resource_type][key][0]) for key in STONE_TYPES]
        for i, text in enumerate(resources):
            self.canvas.blit(self.font.render(text, True, BLACK), self._coords(i))

    @abstractmethod
    def _coords(self, i: int) -> tuple[int, int]:
        raise NotImplementedError()


class StonesIndicators(ResourcesIndicators):
    resource_type = STONES

    def _coords(self, i: int) -> tuple[int, int]:
        return (
            i * self.box_size + self.horizontal_resources_text_offset,
            5 * self.box_size + self.vertical_resources_text_offset,
        )


class RobotsIndicators(ResourcesIndicators):
    resource_type = ROBOTS

    def _coords(self, i: int) -> tuple[int, int]:
        return (
            5 * self.box_size + self.horizontal_resources_text_offset,
            i * self.box_size + self.vertical_resources_text_offset,
        )


class BlueprintIndicators(Indicators):
    def render(self, obs: Observation) -> None:
        ore_costs = [
            str(obs[BLUEPRINT][robot_type][ORE][0]) for robot_type in ROBOT_TYPES
        ]
        for i, text in enumerate(ore_costs):
            self.canvas.blit(
                self.font.render(text, True, BLACK),
                (
                    self.horizontal_resources_text_offset,
                    i * self.box_size + self.vertical_resources_text_offset,
                ),
            )

        obsidian_clay_cost = str(obs[BLUEPRINT][OBSIDIAN][CLAY][0])
        self.canvas.blit(
            self.font.render(obsidian_clay_cost, True, BLACK),
            (
                self.box_size + self.horizontal_resources_text_offset,
                2 * self.box_size + self.vertical_resources_text_offset,
            ),
        )

        geode_obsidian_cost = str(obs[BLUEPRINT][GEODE][OBSIDIAN][0])
        self.canvas.blit(
            self.font.render(geode_obsidian_cost, True, BLACK),
            (
                2 * self.box_size + self.horizontal_resources_text_offset,
                3 * self.box_size + self.vertical_resources_text_offset,
            ),
        )


class TimeIndicators(Indicators):
    def __init__(self, canvas, window_size: int, box_size: int) -> None:
        super().__init__(canvas, window_size, box_size)

        self.horizontal_time_text_offset = 0.3 * self.box_size
        self.vertical_time_text_offset = 0.15 * self.box_size

    def render(self, obs: Observation) -> None:
        current_time = obs[STATE][TIME]
        max_time = obs[MAX_TIME]

        self.canvas.blit(
            self.font.render(f"{current_time[0]}/{max_time[0]}", True, BLACK),
            (
                5 * self.box_size - self.horizontal_time_text_offset,
                5 * self.box_size - self.vertical_time_text_offset,
            ),
        )
