"""Common Utilities"""

import json


class PrettyDict(dict):
    """Dict but it is automatically pretty printed with an indentation."""

    def __str__(self):
        return json.dumps(self, indent=4).__str__()

    def __repr__(self):
        return self.__str__()
