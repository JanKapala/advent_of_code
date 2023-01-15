"""Action and Action Space utilities."""

from gymnasium.spaces import Discrete

from day19.rl.env.constants import ROBOT_TYPES

ACTION_TO_ROBOT_TYPE_MAPPING = ROBOT_TYPES + (None,)

Action = int


def generate_action_space() -> Discrete:
    """Generate action space that is compatible with the `Action` class.

    :return: Action Space.
    """
    return Discrete(len(ACTION_TO_ROBOT_TYPE_MAPPING))
