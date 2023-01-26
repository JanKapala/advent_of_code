"""Action and Action Space utilities."""

from gymnasium.spaces import Discrete

from advent_of_code.day19.env.constants import ROBOT_TYPES

ACTION_TO_ROBOT_TYPE_MAPPING = ROBOT_TYPES + (None,)


class Action(int):
    """Action interface between the Agent and the Environment"""

    # noinspection PyMethodMayBeStatic
    NOOP = -1  # Special Action needed for `gymnasium.utils.play(...)`


def generate_action_space() -> Discrete:
    """Generate action space that is compatible with the `Action` class.

    :return: Action Space.
    """
    return Discrete(len(ACTION_TO_ROBOT_TYPE_MAPPING))
