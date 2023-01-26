from abc import ABC

from pygame.surface import Surface

from advent_of_code.day19.env.observation import Observation


class SceneComponent(ABC):
    def __init__(self, canvas: Surface, window_size: int, box_size: int) -> None:
        self.canvas = canvas
        self.box_size = box_size
        self.window_size = window_size

    def render(self, obs: Observation) -> None:
        raise NotImplementedError()
