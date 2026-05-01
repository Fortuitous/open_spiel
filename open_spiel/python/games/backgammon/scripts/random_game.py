import pyspiel
import random

game = pyspiel.load_game("backgammon")
state = game.new_initial_state()

while not state.is_terminal():
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        action_list, probs = zip(*outcomes)
        action = random.choices(action_list, weights=probs)[0]
    else:
        actions = state.legal_actions()
        action = random.choice(actions)
    state.apply_action(action)

print("Final Board State:")
print(state.to_string())
print("\nReturns:", state.returns())
