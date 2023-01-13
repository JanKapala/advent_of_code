from gymnasium.spaces import Discrete

from day19.rl.env.constants import ROBOT_TYPES

ACTION_TO_ROBOT_TYPE_MAPPING = ROBOT_TYPES + (None, )


def _get_actions_space():
    return Discrete(len(ACTION_TO_ROBOT_TYPE_MAPPING))
