import pyspiel

game = pyspiel.load_game("backgammon")
state = game.new_initial_state()

for action, prob in state.chance_outcomes():
    state_copy = state.clone()
    state_copy.apply_action(action)
    if "21" in str(state_copy) or "12" in str(state_copy):
        state.apply_action(action)
        break

def print_plane0(state, title):
    print(f"=== {title} ===")
    print("State:\n" + state.to_string())
    try:
        obs = state.observation_tensor()
        plane0 = obs[0:24]
        row_top = plane0[12:24]
        row_bottom = plane0[0:12][::-1]
        print("--- PLANE 0: My Checker Presence ---")
        print("  " + " ".join([f"{v:.4f}" for v in row_top]))
        print("  " + " ".join([f"{v:.4f}" for v in row_bottom]))
    except Exception as e:
        print("Could not retrieve observation tensor:", e)
    print()

print_plane0(state, "Initial Opening Position (2-1) [Player " + str(state.current_player()) + " to move]")

print("Legal Actions:")
target_action = None
for a in state.legal_actions():
    action_str = state.action_to_string(state.current_player(), a)
    print(f"Action ID: {a} -> {action_str}")
    if "13/11 6/5" in action_str or "6/5 13/11" in action_str:
        target_action = a
print()

if target_action is not None:
    print(f"Executing Action ID {target_action} ('13/11 6/5')...\n")
    state.apply_action(target_action)
else:
    print("Could not find '13/11 6/5'. Exiting.")
    exit(1)

if state.is_chance_node():
    print("State is now at a CHANCE node. Applying a 3-3 roll to reach the next player's Decision node...\n")
    for action, prob in state.chance_outcomes():
        state_copy = state.clone()
        state_copy.apply_action(action)
        if "33" in str(state_copy):
            state.apply_action(action)
            break

print_plane0(state, "After 13/11 6/5 [Player " + str(state.current_player()) + " to evaluate]")
