import pyspiel

def print_audit(state, filename):
    obs = state.observation_tensor()
    tensor_size = len(obs)
    
    with open(filename, "w") as f:
        f.write(f"=== FULL TENSOR AUDIT ===\n")
        f.write("State:\n")
        f.write(state.to_string() + "\n\n")
        f.write(f"Total Vector Size: {tensor_size}\n\n")
        
        f.write("--- ALL GLOBAL SCALARS ---\n")
        labels = [
            "My Score (Born-off/15)", "Opp Score (Born-off/15)",
            "My Moves Remaining (N/4.0)", "Opp Moves Remaining (N/4.0)",
            "My Pips (N/375)", "Opp Pips (N/375)", "Pip Difference (N/375)",
            "Contact", "My Home Points Made (N/6.0)", "My Bar (Count/15)",
            "My Bar == 1", "My Bar == 2", "My Bar == 3", "My Bar >= 4",
            "Opp Home Points Made (N/6.0)", "Opp Bar (Count/15)",
            "Opp Bar == 1", "Opp Bar == 2", "Opp Bar == 3", "Opp Bar >= 4"
        ]
        
        scalar_start = 1200 if tensor_size == 1220 else 1080
        multipliers = [15, 15, 4, 4, 375, 375, 375, 1, 6, 15, 1, 1, 1, 1, 6, 15, 1, 1, 1, 1]
        
        for i, label in enumerate(labels):
            idx = scalar_start + i
            if idx < tensor_size:
                base_str = f"[{idx}] {label:<28}: {obs[idx]:.6f}"
                if i < len(multipliers):
                    raw_val = int(round(obs[idx] * multipliers[i]))
                    f.write(f"{base_str}  ({raw_val})\n")
                else:
                    f.write(f"{base_str}\n")
        
        f.write("\n")
        plane_descriptions = [
            "My Checker Presence", "Opp Checker Presence",
            "My Checkers == 1", "My Checkers == 2", "My Checkers == 3", 
            "My Checkers == 4", "My Checkers == 5", "My Checkers >= 6",
            "Opp Checkers == 1", "Opp Checkers == 2", "Opp Checkers == 3", 
            "Opp Checkers == 4", "Opp Checkers == 5", "Opp Checkers >= 6",
            "My Score (Born-off/15)", "Opp Score (Born-off/15)",
            "My Moves Remaining (N/4.0)", "Opp Moves Remaining (N/4.0)",
            "My Pips (N/375)", "Opp Pips (N/375)", "Pip Difference (N/375)",
            "Contact",
            "My Deep Anchors", "My Advanced Anchors",
            "My Primes == 2", "My Primes == 3", "My Primes == 4", "My Primes == 5", "My Primes >= 6",
            "My Blockade Density", "My Home Board Strength",
            "My Bar == 1", "My Bar == 2", "My Bar == 3", "My Bar >= 4",
            "Opp Deep Anchors", "Opp Advanced Anchors",
            "Opp Primes == 2", "Opp Primes == 3", "Opp Primes == 4", "Opp Primes == 5", "Opp Primes >= 6",
            "Opp Blockade Density", "Opp Home Board Strength",
            "Opp Bar == 1", "Opp Bar == 2", "Opp Bar == 3", "Opp Bar >= 4",
            "Reserved / Padding (All Zeros)", "Reserved / Padding (All Zeros)"
        ]
        
        num_planes = tensor_size // 24
        for p in range(num_planes):
            desc = plane_descriptions[p] if p < len(plane_descriptions) else "Unknown"
            f.write(f"--- PLANE {p}: {desc} ---\n")
            plane = obs[p*24 : (p+1)*24]
            row_top = plane[12:24]
            row_bottom = plane[0:12][::-1]
            f.write("  " + " ".join([f"{v:.4f}" for v in row_top]) + "\n")
            f.write("  " + " ".join([f"{v:.4f}" for v in row_bottom]) + "\n\n")

game = pyspiel.load_game("backgammon")
state = game.new_initial_state()

for action, prob in state.chance_outcomes():
    state_copy = state.clone()
    state_copy.apply_action(action)
    if "42" in str(state_copy) or "24" in str(state_copy):
        state.apply_action(action)
        break

# Play 8/4 6/4 (4-2)
# We found earlier this is action 456437
try:
    state.apply_action(456437)
except:
    state.apply_action(state.legal_actions()[0])

for action, prob in state.chance_outcomes():
    state_copy = state.clone()
    state_copy.apply_action(action)
    if "33" in str(state_copy):
        state.apply_action(action)
        break

# Audit 1: Board after opening 4-2, from perspective of opponent rolling 3-3
print_audit(state, "audit_Move1.txt")

# Play a 3-3 response
state.apply_action(state.legal_actions()[0])

for action, prob in state.chance_outcomes():
    state_copy = state.clone()
    state_copy.apply_action(action)
    if "21" in str(state_copy) or "12" in str(state_copy):
        state.apply_action(action)
        break

# Audit 2: Board after 4-2 and 3-3, from perspective of Player 1 rolling 2-1
print_audit(state, "audit_Move2.txt")
