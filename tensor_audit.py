import sys
import pyspiel
import numpy as np

def audit_tensor(xgid):
    game = pyspiel.load_game("backgammon(dmp_only=True)")
    state = game.new_initial_state()
    
    # Simple XGID parser for common fields
    parts = xgid.split(':')
    board_str = parts[0].replace('XGID=', '')
    turn_str = parts[3]
    dice_str = parts[4]
    
    board = [[0]*24 for _ in range(2)]
    # In XGID, Uppercase is the Bottom Player (OpenSpiel Player 1 / o)
    # Lowercase is the Top Player (OpenSpiel Player 0 / x)
    p0_chars = "abcdefghijklmnopqrstuvwxyz" # Top Player
    p1_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # Bottom Player
    
    def get_count(char):
        if char in p0_chars: return p0_chars.index(char) + 1
        if char in p1_chars: return p1_chars.index(char) + 1
        return 0

    # Index 0 is Top Player's (P0) Bar. Index 25 is Bottom Player's (P1) Bar.
    bar = [0, 0]
    if len(board_str) == 26:
        bar[0] = get_count(board_str[0])
        bar[1] = get_count(board_str[25])
    
    # Index 1-24 are the board points from Bottom Player's perspective (1 to 24).
    # OpenSpiel pos 0 is Bottom Player's 1-point. So pos = i - 1.
    for i in range(1, min(25, len(board_str))):
        char = board_str[i]
        pos = i - 1
        if char in p0_chars:
            board[0][pos] = p0_chars.index(char) + 1
        elif char in p1_chars:
            board[1][pos] = p1_chars.index(char) + 1
            
    # In XGID, 1 is Bottom Player's turn (P1). -1 is Top Player's turn (P0).
    turn = 1 if turn_str == "1" else 0
    dice = [int(d) for d in dice_str if d != '0']
    
    # Calculate scores (borne off checkers) to satisfy SPIEL_CHECKs (15 checkers total)
    scores = [0, 0]
    for p in range(2):
        total_on_board_and_bar = sum(board[p]) + bar[p]
        if total_on_board_and_bar < 15:
            scores[p] = 15 - total_on_board_and_bar
            
    state.set_state(turn, False, dice, bar, scores, board)
    
    obs = state.observation_tensor()
    tensor_size = len(obs)
    
    filename = f"audit_{xgid.replace(':', '_').replace('=', '_')}.txt"
    display_xgid = xgid if xgid.startswith("XGID=") else f"XGID={xgid}"
    with open(filename, "w") as f:
        f.write(f"=== FULL TENSOR AUDIT v12.2: {display_xgid} ===\n")
        f.write("State:\n")
        f.write(state.to_string() + "\n\n")
        f.write(f"Total Vector Size: {tensor_size}\n\n")
        
        f.write("--- ALL GLOBAL SCALARS ---\n")
        # v12.2 Scalar Start is 1200
        labels = [
            "My Score (Born-off/15)", "Opp Score (Born-off/15)",
            "My Moves Remaining (N/4.0)", "Opp Moves Remaining (N/4.0)",
            "My Pips (N/375)", "Opp Pips (N/375)", "Pip Difference (N/375)",
            "Contact", "My Home Points Made (N/6.0)", "My Bar (Count/15)",
            "My Bar == 1", "My Bar == 2", "My Bar == 3", "My Bar >= 4",
            "Opp Home Points Made (N/6.0)", "Opp Bar (Count/15)",
            "Opp Bar == 1", "Opp Bar == 2", "Opp Bar == 3", "Opp Bar >= 4"
        ]
        
        # Determine scalar start based on vector size
        scalar_start = 1200 if tensor_size == 1220 else 1080
        
        multipliers = [
            15, 15, 4, 4, 375, 375, 375, 1, 
            6, 15, 1, 1, 1, 1,
            6, 15, 1, 1, 1, 1
        ]
        
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
            
    print(f"Audit complete. Results in {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tensor_audit.py <XGID>")
    else:
        audit_tensor(sys.argv[1])
