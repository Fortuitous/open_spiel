import pyspiel

game = pyspiel.load_game("backgammon")
state = game.new_initial_state()

for action, prob in state.chance_outcomes():
    state_copy = state.clone()
    state_copy.apply_action(action)
    if "21" in str(state_copy) or "12" in str(state_copy):
        state.apply_action(action)
        break

target_action = None
for a in state.legal_actions():
    action_str = state.action_to_string(state.current_player(), a)
    if "13/11 6/5" in action_str or "6/5 13/11" in action_str:
        target_action = a
        break

if target_action is not None:
    state.apply_action(target_action)

if state.is_chance_node():
    for action, prob in state.chance_outcomes():
        state_copy = state.clone()
        state_copy.apply_action(action)
        if "33" in str(state_copy):
            state.apply_action(action)
            break

def print_plane1(state, title):
    print(f"=== {title} ===")
    print("State:\n" + state.to_string())
    try:
        obs = state.observation_tensor()
        plane1 = obs[24:48]
        row_top = plane1[12:24]
        row_bottom = plane1[0:12][::-1]
        print("--- PLANE 1: Opp Checker Presence ---")
        print("  " + " ".join([f"{v:.4f}" for v in row_top]))
        print("  " + " ".join([f"{v:.4f}" for v in row_bottom]))
    except Exception as e:
        print("Could not retrieve observation tensor:", e)
    print()

print_plane1(state, "After 13/11 6/5 [Player " + str(state.current_player()) + " evaluating their 3-3]")
