from data_loading import load_blueprints, INPUT_FILE_PATH
from state import State
from memory import Memory
from walker import walker

blueprints = load_blueprints(INPUT_FILE_PATH)
for blueprint in blueprints:
    memory = Memory()
    memory.put(State(blueprint=blueprint))
    best_score = walker(memory)
    break  # TODO: delete this