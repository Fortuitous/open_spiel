import sys
sys.path.append("/home/jeremy/projects/open_spiel/build/python")

import torch
import pyspiel
import numpy as np
from expert_eyes_model import ExpertEyesNet

def clean_dice_for_xg(dice_str):
    """Converts '(roll: 12)' to sorted '21:' for XG compatibility."""
    if "roll: " in dice_str:
        raw = dice_str.split("roll: ")[1].strip(")")
        sorted_dice = "".join(sorted(list(raw), reverse=True))
        return sorted_dice + ":"
    return dice_str + ":"

def format_moves_for_xg(move_str):
    """
    Surgically removes OpenSpiel Action IDs and sanitizes 'Pass' for XG.
    """
    # 1. Robustly strip the Action ID (e.g., '456300 - 24/23' -> '24/23')
    if " - " in move_str:
        move_str = move_str.split(" - ")[-1].strip()
    elif " / " in move_str: # Just in case
        move_str = move_str.split(" / ")[-1].strip()
    
    # 2. Handle 'Pass' cases
    if move_str == "Pass":
        return "Cannot Move"
    
    # Partial pass: if 'Pass' is at the end or embedded:
    if "Pass" in move_str:
        move_str = move_str.replace("Pass", "").strip()
    
    # 3. Standard XG board notation
    return move_str.replace("-", "/").replace(",", " ")

def run_smoke_test_game():
    game = pyspiel.load_game("backgammon(dmp_only=True)")
    model = ExpertEyesNet(num_res_blocks=12, num_filters=256)
    model.eval()
    
    state = game.new_initial_state()
    move_records = [] 
    current_turn_dice = ""

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
            current_turn_dice = clean_dice_for_xg(state.action_to_string(action))
            state.apply_action(action)
        else:
            player = state.current_player()
            obs_tensor = state.observation_tensor()
            tensor_input = torch.FloatTensor(obs_tensor).unsqueeze(0)
            
            with torch.no_grad():
                _, value = model(tensor_input)
            
            legal_actions = state.legal_actions()
            action = legal_actions[0] 
            move_str = format_moves_for_xg(state.action_to_string(action))
            
            move_records.append({
                "player": player,
                "dice": current_turn_dice,
                "move": move_str,
                "value": value.item()
            })
            state.apply_action(action)

    # --- SAVE XG-COMPATIBLE .TXT LOG ---
    with open("smoke_test_xg.txt", "w") as f:
        f.write("; [Site \"Expert Eyes v12.1 Engine\"]\n")
        f.write("; [Variation \"Backgammon\"]\n\n")
        f.write("1 point match\n\n\n")
        f.write(" Game 1\n")
        f.write(" Player 1 : 0                         Player 2 : 0\n")
        
        for i in range(0, len(move_records), 2):
            move_num = (i // 2) + 1
            p1 = move_records[i]
            p2 = move_records[i+1] if i+1 < len(move_records) else None
            
            # Using 28 chars for spacing to perfectly align the columns
            line = f"  {move_num}) {p1['dice']} {p1['move']:<28}"
            if p2:
                line += f"{p2['dice']} {p2['move']}"
            f.write(line + "\n")
            
        f.write(f"  Wins 1 point\n")

    # --- SAVE FULL ANALYSIS LOG ---
    with open("smoke_test_analysis.txt", "w") as f:
        f.write("v12.1 Expert Eyes Analysis Log (DMP)\n")
        f.write("-" * 50 + "\n")
        for i, m in enumerate(move_records):
            f.write(f"Move {i+1:03} | Player {m['player']} | {m['dice']} {m['move']}\n")
            f.write(f"Value Prediction (Self Win Prob): {m['value']:.6f}\n\n")

if __name__ == "__main__":
    run_smoke_test_game()
