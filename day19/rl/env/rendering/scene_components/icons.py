import os
from abc import abstractmethod

import pygame

from day19.rl.env.constants import STONE_TYPES, ASSETS_PATH
from day19.rl.env.observation import Observation
from day19.rl.env.rendering.scene_components.scene_component import SceneComponent


class Icon(SceneComponent):
    icon_scale = 0.8
    icon_offset = (1 - icon_scale) / 2

    def _transform_icon(self, icon):
        return pygame.transform.scale(
            icon, (self.box_size * self.icon_scale, self.box_size * self.icon_scale)
        )

    @abstractmethod
    def _coords(self, i) -> tuple[int, int]:
        raise NotImplementedError()


class ResourcesIcons(Icon):  # TODO: change name to ResourcesIcons
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


class RobotsIcons(ResourcesIcons):
    name_suffix = "_robot_icon.png"
    icons_subdir_name = "robots"

    def _coords(self, i):
        return (4 + self.icon_offset) * self.box_size, (
            i + self.icon_offset
        ) * self.box_size


class StonesIcons(ResourcesIcons):  # TODO: change name to StonesIcons
    name_suffix = "_stone_icon.png"
    icons_subdir_name = "resources"

    def _coords(self, i):
        return (i + self.icon_offset) * self.box_size, (
            4 + self.icon_offset
        ) * self.box_size