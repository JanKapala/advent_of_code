import random
from collections import deque
from typing import Union

import torch
from torch.nn.utils.rnn import pad_sequence

DONE = "done"


class ReplayBuffer:
    """Replay Buffer used in the TD3 Agent"""

    def __init__(
        self,
        batch_size: int,
        max_size: int,
    ) -> None:
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_size)

    @property
    def max_size(self):
        return self.buffer.maxlen

    @max_size.setter
    def max_size(self, new_value):
        self.buffer = deque(self.buffer, maxlen=new_value)

    def __iter__(self):
        return self

    def __next__(self):
        batch_size = min(self.batch_size, len(self.buffer))
        if batch_size == 0:
            raise Exception("Replay Buffer is Empty")

        raw_records = random.sample(self.buffer, k=batch_size)



        return batch

    def _add_record(self, record):
        assert len(record) == 5
        state, weights, reward, next_state, done = record

        user, services_history, mask = self.state_encoder([state, next_state])
        reward = self.reward_encoder([reward])

        example = {
            STATE: ...,
            ACTION: weights[0],
            REWARD: reward[0],
            NEXT_STATE: ...,
            DONE: torch.tensor(float(done)),
        }

        self.buffer.append(example)

    def __lshift__(self, record):
        self._add_record(record)

    def __rshift__(self, record):
        self._add_record(record)

    def __len__(self):
        if self.buffer:
            return self.buffer[REWARD].shape[0]
        return 0