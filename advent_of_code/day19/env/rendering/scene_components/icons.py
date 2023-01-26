# pylint: disable=too-few-public-methods

"""Icons needed for the rendering of the environment scene."""

import os
from abc import ABC, abstractmethod

import pygame

from advent_of_code.day19.env.constants import ASSETS_PATH, STONE_TYPES
from advent_of_code.day19.env.observation import Observation
from advent_of_code.day19.env.rendering.scene_components.scene_component import (
    SceneComponent,
)


class Icon(SceneComponent, ABC):
    """Image icon that can be rendered using pygame."""

    icon_scale = 0.8
    icon_offset = (1 - icon_scale) / 2

    def _transform_icon(self, icon):
        return pygame.transform.scale(
            icon, (self.box_size * self.icon_scale, self.box_size * self.icon_scale)
        )


class ResourcesIcons(Icon, ABC):
    """Row of image icons that can be rendered using pygame."""

    name_suffix = None
    icons_subdir_name = None

    def __init__(self, canvas, window_size: int, box_size: int) -> None:
        super().__init__(canvas, window_size, box_size)

        self.icons = []
        self.types = STONE_TYPES

        for img_name in [name_prefix + self.name_suffix for name_prefix in self.types]:
            icon = pygame.image.load(
                os.path.join(ASSETS_PATH, self.icons_subdir_name, img_name)
            )
            self.icons.append(self._transform_icon(icon))

    def render(self, _obs: Observation) -> None:
        for i in range(4):
            rect = self.icons[i].get_rect()
            rect = rect.move(self._coords(i))
            self.canvas.blit(self.icons[i], rect)

    @abstractmethod
    def _coords(self, i) -> tuple[int, int]:
        raise NotImplementedError()


class RobotsIcons(ResourcesIcons):
    """Row of robots icons that can be rendered using pygame."""

    name_suffix = "_robot_icon.png"
    icons_subdir_name = "robots"

    def _coords(self, i):
        return (4 + self.icon_offset) * self.box_size, (
            i + self.icon_offset
        ) * self.box_size


class StonesIcons(ResourcesIcons):
    """Row of stones icons that can be rendered using pygame."""

    name_suffix = "_stone_icon.png"
    icons_subdir_name = "resources"

    def _coords(self, i):
        return (i + self.icon_offset) * self.box_size, (
            4 + self.icon_offset
        ) * self.box_size
