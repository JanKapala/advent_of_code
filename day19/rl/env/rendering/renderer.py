# pylint: disable-all
# TODO: pylint enable all.

from time import sleep, time

import numpy as np
import pygame

# TODO: apply naming hierarchy resources: stones, robots in the whole project
# TODO: suppress pygame log
from numpy import ndarray

from day19.constants import DAY_19_INPUT_FILE_PATH
from day19.rl.data_loading import load_blueprints, extract_global_data
from day19.rl.env.observation import Observation
from day19.rl.env.rendering.colors import WHITE
from day19.rl.env.rendering.scene_components.grid import Grid
from day19.rl.env.rendering.scene_components.icons import RobotsIcons, StonesIcons
from day19.rl.env.rendering.scene_components.indicators import (
    StonesIndicators,
    RobotsIndicators,
    BlueprintIndicators,
    TimeIndicators,
)
from day19.rl.env.state import State


class Renderer:
    _components_classes = (
        Grid,
        RobotsIcons,
        StonesIcons,
        RobotsIndicators,
        StonesIndicators,
        BlueprintIndicators,
        TimeIndicators,
    )

    def __init__(self, render_mode, window_size=500, render_fps=4):
        if render_mode is False:
            return  # TODO: do it better
        self.render_mode = render_mode
        self.window_size = window_size
        self.box_size = int(self.window_size / 6)

        self.render_fps = render_fps  # TODO: experiment with this value

        # TODO: maybe do it only if render=True in the Env
        pygame.init()
        pygame.display.init()

        self.window = pygame.display.set_mode(size=(window_size, window_size))
        pygame.display.set_caption("Not Enough Minerals")

        self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface(size=(self.window_size, self.window_size))

        #  TODO: vectorize this
        self._components = [
            clazz(self.canvas, self.window_size, self.box_size)
            for clazz in self._components_classes
        ]

    def render(self, obs: Observation) -> None | ndarray:
        self.canvas.fill(color=WHITE)

        for component in self._components:
            component.render(obs)

        if self.render_mode is None:
            pass
        elif self.render_mode == "human":
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
            )

        self.clock.tick(self.render_fps)

    def close(self):
        pygame.display.quit()
        pygame.quit()


if __name__ == "__main__":
    start = time()
    blueprints = load_blueprints(DAY_19_INPUT_FILE_PATH)
    robots_costs_boundaries, costs_boundaries = extract_global_data(blueprints)

    end = time()
    print(f"Data loading time: {end - start}")

    obs = Observation(state=State(), blueprint=blueprints[0], max_time=24)
    start = time()
    renderer = Renderer(window_size=500, render_fps=4)
    end = time()
    print(f"Renderer initialization time: {end-start}")

    start = time()
    renderer.render(obs)
    end = time()
    print(f"Frame rendering time: {end - start}")

    start = time()
    renderer.close()
    end = time()
    print(f"Renderer closing time: {end - start}")

    sleep(5)
