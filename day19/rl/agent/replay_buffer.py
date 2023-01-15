# pylint: disable=no-member, invalid-name, fixme

"""Replay Buffer used in the TD3 Agent"""

import random
from collections import deque
from typing import Deque, TypeAlias, cast

import torch
from torch import Tensor

OBSERVATION = "observation"
ACTION = "action"
REWARD = "reward"
NEXT_OBSERVATION = "next_observation"
DONE = "done"


Record: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

# TODO: generally do something with pylint `no-member` and `no-name-in-module`
#  offences everything in pytorch causes it xD


class ReplayBuffer:
    """Replay Buffer used in the TD3 Agent"""

    def __init__(
        self,
        batch_size: int,
        max_size: int,
    ) -> None:
        self.batch_size = batch_size
        self.buffer: Deque = deque(maxlen=max_size)

    @property
    def max_size(self) -> int:  # pylint: disable=missing-function-docstring
        return cast(int, self.buffer.maxlen)

    @max_size.setter
    def max_size(self, new_value: int) -> None:
        self.buffer = deque(self.buffer, maxlen=new_value)

    def __iter__(self) -> "ReplayBuffer":
        return self

    def __next__(self) -> Record:
        batch_size = min(self.batch_size, len(self.buffer))
        if batch_size == 0:
            raise Exception("Replay Buffer is Empty")

        raw_batch = random.sample(self.buffer, k=batch_size)

        names = [OBSERVATION, ACTION, REWARD, NEXT_OBSERVATION, DONE]
        result = tuple(
            torch.concat([example[key] for example in raw_batch], dim=0) for key in names
        )
        return cast(Record, result)

    def _add_record(self, record: Record):
        assert len(record) == 5
        O, A, R, O_prim, T = record

        example = {
            OBSERVATION: O,
            ACTION: A,
            REWARD: R,
            NEXT_OBSERVATION: O_prim,
            DONE: T,
        }

        self.buffer.append(example)

    def __lshift__(self, record):
        self._add_record(record)

    def __len__(self):
        if self.buffer:
            return self.buffer[REWARD].shape[0]
        return 0
