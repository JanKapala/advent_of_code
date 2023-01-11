import random

import pandas as pd
from IPython.core.display_functions import display
from torch.utils.tensorboard import SummaryWriter

from day19.actions import Action
from memory import Memory
from blueprint import Blueprint

MAX_TIME = 24  # TODO: do it better

writer = SummaryWriter(log_dir="./runs")


def walker(memory: Memory):
    best_score = 0
    processed_states = 0
    while not memory.empty():
        state = memory.get()

        writer.add_scalar("memory size", len(memory), processed_states)
        writer.flush()

        if state.resources.geode > best_score:
            best_score = state.resources.geode

            writer.add_scalar("best score", best_score, processed_states)
            writer.flush()

            display(state.history_dataframe)

        if state.time == MAX_TIME:  # TODO: make sure if this is a correct condition
            continue

        # TODO: take max costs into consideration in the states comparisons
        # TODO: implement rest of comparison operators in the states
        actions = list(Action)
        # random.shuffle(actions)  # TODO: test if it make a difference

        for action in actions:
            memory.put(state.next(action))

        processed_states += 1

    return best_score