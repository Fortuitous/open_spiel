import sys
sys.path.append("/home/jeremy/projects/open_spiel")
sys.path.append("/home/jeremy/projects/open_spiel/build/python")

import torch
import pyspiel
import numpy as np
from open_spiel.python.algorithms import mcts
from expert_eyes_model import ExpertEyesNet

def clean_dice_for_xg(dice_str):
    if "roll: " in dice_str:
        raw = dice_str.split("roll: ")[1].strip(")")
        sorted_dice = "".join(sorted(list(raw), reverse=True))
        return sorted_dice + ":"
    return dice_str + ":"

def format_moves_for_xg(move_str):
    if " - " in move_str:
        move_str = move_str.split(" - ")[-1].strip()
    elif " / " in move_str:
        move_str = move_str.split(" / ")[-1].strip()
        
    if move_str == "Pass":
        return "Cannot Move"
        
    if "Pass" in move_str:
        move_str = move_str.replace("Pass", "").strip()
        
    return move_str.replace("-", "/").replace(",", " ")

class ExpertEyesEvaluator(mcts.Evaluator):
    def __init__(self, model):
        self.model = model

    def evaluate(self, state):
        if state.is_chance_node():
            return np.zeros(2)
            
        obs_tensor = state.observation_tensor()
        tensor_input = torch.FloatTensor(obs_tensor).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(tensor_input)
            
        v = value.item()
        curr_p = state.current_player()
        
        returns = np.zeros(2)
        if curr_p >= 0:
            returns[curr_p] = v
            returns[1 - curr_p] = -v
        return returns

    def prior(self, state):
        if state.is_chance_node():
            return state.chance_outcomes()
        legal_actions = state.legal_actions()
        return [(action, 1.0 / len(legal_actions)) for action in legal_actions]

def run_self_play_v1():
    game = pyspiel.load_game("backgammon(dmp_only=True)")
    model = ExpertEyesNet(num_res_blocks=12, num_filters=256)
    model.eval()
    
    evaluator = ExpertEyesEvaluator(model)
    mcts_bot = mcts.MCTSBot(
        game,
        uct_c=1.1,
        max_simulations=40,
        evaluator=evaluator,
        solve=True
    )
    
    state = game.new_initial_state()
    move_records = [] 
    current_turn_dice = ""
    analysis_lines = []

    print(f"--- STARTING MCTS SELF-PLAY GAME ---")
    
    step_idx = 0
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = np.random.choice(actions, p=probs)
            current_turn_dice = clean_dice_for_xg(state.action_to_string(action))
            state.apply_action(action)
        else:
            player = state.current_player()
            
            # Step the MCTS bot to get the root tree
            root = mcts_bot.mcts_search(state)
            
            # Extract top actions from children
            children = root.children
            # Sort by explore_count descending
            sorted_children = sorted(children, key=lambda c: c.explore_count, reverse=True)
            
            action = sorted_children[0].action
            move_str = format_moves_for_xg(state.action_to_string(player, action))
            
            move_records.append({
                "player": player,
                "dice": current_turn_dice,
                "move": move_str
            })
            
            step_idx += 1
            print(f"Step {step_idx:03} | P{player} | {current_turn_dice} {move_str} (MCTS sims: {root.explore_count})")
            
            # Generate Analysis Logging String
            anal_str = f"Move {step_idx:03} | Player {player} | {current_turn_dice} {move_str}\n"
            anal_str += f"Root Visit Count: {root.explore_count}\n"
            for i, child in enumerate(sorted_children[:3]):
                c_move = format_moves_for_xg(state.action_to_string(player, child.action))
                avg_val = (child.total_reward / child.explore_count) if child.explore_count > 0 else 0.0
                anal_str += f"  [{i+1}] {c_move:<20} | Visits: {child.explore_count:2} | Value: {avg_val:+.4f}\n"
            anal_str += "\n"
            analysis_lines.append(anal_str)
            
            state.apply_action(action)

    # --- SAVE XG-COMPATIBLE .TXT LOG ---
    with open("mcts_self_play_xg.txt", "w") as f:
        f.write("; [Site \"Expert Eyes MCTS v1.1\"]\n")
        f.write("; [Variation \"Backgammon\"]\n\n")
        f.write("1 point match\n\n\n")
        f.write(" Game 1\n")
        f.write(" Player 1 : 0                         Player 2 : 0\n")
        
        for i in range(0, len(move_records), 2):
            move_num = (i // 2) + 1
            p1 = move_records[i]
            p2 = move_records[i+1] if i+1 < len(move_records) else None
            
            line = f"  {move_num}) {p1['dice']} {p1['move']:<28}"
            if p2:
                line += f"{p2['dice']} {p2['move']}"
            f.write(line + "\n")
            
        f.write(f"  Wins 1 point\n")

    # --- SAVE MCTS ANALYSIS LOG ---
    with open("mcts_analysis.txt", "w") as f:
        f.write("v1.1 MCTS Self-Play Analysis Log (40 Sims/Move)\n")
        f.write("-" * 60 + "\n")
        for line in analysis_lines:
            f.write(line)

    print(f"--- SUCCESS: MCTS Pipeline Executed ---")

if __name__ == "__main__":
    run_self_play_v1()
