import pygame

from day19.rl.env.observation import Observation
from day19.rl.env.rendering.scene_components.scene_component import SceneComponent


class Grid(SceneComponent):
    def render(self, _obs: Observation):
        for i in range(4):
            pygame.draw.line(
                self.canvas,
                0,
                (0, (i + 1) * self.box_size),
                (self.window_size, (i + 1) * self.box_size),
                width=3,
            )
            pygame.draw.line(
                self.canvas,
                0,
                ((i + 1) * self.box_size, 0),
                ((i + 1) * self.box_size, self.window_size),
                width=3,
            )
        pygame.draw.line(
            self.canvas,
            0,
            (0, 5 * self.box_size),
            (self.window_size - 2 * self.box_size, 5 * self.box_size),
            width=3,
        )
        pygame.draw.line(
            self.canvas,
            0,
            (5 * self.box_size, 0),
            (5 * self.box_size, self.window_size - 2 * self.box_size),
            width=3,
        )
