# pylint: disable-all
# TODO: pylint enable all.

import os
from copy import deepcopy
from time import sleep

import numpy as np
import pygame

# TODO: suppress pygame log
from day19.constants import DAY_19_ROOT_DIR

ASSETS_PATH = os.path.join(DAY_19_ROOT_DIR, "rl/env/rendering/assets")

if __name__ == "__main__":
    pygame.init()
    pygame.display.init()

    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 128)
    BLACK = (0, 0, 0)

    window_height = 500

    box_size = int(0.2 * window_height)
    margin_size = int(0.2 / 5 * window_height)

    window_width = int(3 * box_size + 4 * margin_size)
    canvas_color = WHITE
    box_color = BLACK

    window = pygame.display.set_mode(size=(window_width, window_height))
    pygame.display.set_caption("Not Enough Minerals")

    clock = pygame.time.Clock()

    canvas = pygame.Surface(size=(window_width, window_height))
    canvas.fill(color=canvas_color)

    images = []
    for img_name in ["ore.png", "clay.png", "obsidian.png", "geode.png"]:
        img = pygame.image.load(os.path.join(ASSETS_PATH, img_name))
        images.append(pygame.transform.scale(img, (box_size, box_size)))

    font1 = pygame.font.SysFont("chalkduster.ttf", int(box_size * 0.8))
    img1 = font1.render("XD", True, WHITE)

    for i in range(4):

        rect = images[i].get_rect()
        rect = rect.move(
            (
                margin_size,
                margin_size + i * (box_size + margin_size),
            )
        )
        canvas.blit(images[i], rect)

        for j in range(2):
            # textRect = text.get_rect()
            # textRect.center = (
            #     margin_size+j*(box_size+margin_size),
            #     margin_size+i*(box_size+margin_size)
            # )

            pygame.draw.rect(
                canvas,
                box_color,
                pygame.Rect(
                    (
                        box_size + 2 * margin_size + j * (box_size + margin_size),
                        margin_size + i * (box_size + margin_size),
                    ),
                    (box_size, box_size),
                ),
            )
            canvas.blit(
                img1,
                (
                    box_size
                    + 2 * margin_size
                    + j * (box_size + margin_size)
                    + box_size * 0.1,
                    margin_size + i * (box_size + margin_size) + box_size * 0.25,
                ),
            )

    window.blit(canvas, canvas.get_rect())
    pygame.event.pump()
    pygame.display.update()
    render_fps = 4  # TODO: experiment with this value
    clock.tick(render_fps)

    sleep(5)

    pygame.display.quit()
    pygame.quit()
