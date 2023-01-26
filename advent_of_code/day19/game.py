from gymnasium.utils.play import play

from constants import DAY_19_INPUT_FILE_PATH
from advent_of_code.day19.data_loading import load_blueprints, extract_global_data
from advent_of_code.day19.env.action import Action
from advent_of_code.day19.env.environment import NotEnoughMineralsEnv


if __name__ == "__main__":
    blueprints = load_blueprints(DAY_19_INPUT_FILE_PATH)
    robots_costs_boundaries, costs_boundaries = extract_global_data(blueprints)

    MAX_TIME = 24
    environment = NotEnoughMineralsEnv(
        blueprint=blueprints[0],
        max_time=MAX_TIME,
        robots_costs_boundaries=robots_costs_boundaries,
        render_mode="rgb_array",
    )

    play(environment, noop=Action.NOOP)

    # TODO: save_video handling
