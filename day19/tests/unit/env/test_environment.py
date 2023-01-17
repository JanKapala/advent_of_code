# TODO
from gymnasium.utils.env_checker import check_env

from day19.constants import DAY_19_INPUT_FILE_PATH
from day19.rl.data_loading import load_blueprints, extract_global_data
from day19.rl.env.environment import NotEnoughMineralsEnv


def test_environment():
    blueprints = load_blueprints(DAY_19_INPUT_FILE_PATH)
    robots_costs_boundaries, costs_boundaries = extract_global_data(blueprints)

    max_time = 24

    environment = NotEnoughMineralsEnv(
        blueprint=blueprints[0],
        max_time=max_time,
        robots_costs_boundaries=robots_costs_boundaries,
    )

    obs, _ = environment.reset()
    environment.render()

    print(check_env(environment))  # TODO: Implement this test properly and as a proper test type
