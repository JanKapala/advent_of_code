from random import randint

from state import State
from random import choices

A = 10

class Memory:
    def __init__(self) -> None:
        self.queue: list[State] = []

    def put(self, state: State) -> None:
        if state is None:  # In case when action takes no effect and doesn't produce state
            return

        states_indices_to_remove = []
        for i, other in enumerate(self.queue):
            if state < other or state == other:
                return
            if state > other:
                states_indices_to_remove.append(i)

        for i in sorted(states_indices_to_remove, reverse=True):
            del self.queue[i]

        self.queue.append(state)

    def get(self) -> State:
        # if not self.empty():
        #     result, self.queue = self.queue[0], self.queue[1:]
        #     return result
        if not self.empty():
            i = choices(
                population=range(len(self)),
                weights=[1 + A * state.resources.geode for state in self.queue],
                k=1
            )[0]
            result = self.queue[i]
            self.queue.pop(i)
            return result
        raise BufferError("Memory is empty!")

    def empty(self) -> bool:
        return len(self.queue) == 0

    def __len__(self):
        return len(self.queue)


# TODO: tests
# m = Memory()
# assert m.empty()
#
# m.put(1)
# assert m.get() == 1
# assert m.empty()
#
# m.put(2)
# m.put(3)
# m.put(2)
# m.put(4)
#
# # assert m.empty() is False
# # assert len(m.queue) == 3
# # assert m.get() == 2
# # assert m.get() == 3
# # assert m.get() == 4