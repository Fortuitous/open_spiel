import sys
import numpy as np

# Adjust pythonpath to find pyspiel library from build directory
sys.path.append('/home/jeremy/projects/open_spiel/build/python')
import pyspiel

try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, headers):
        res = " | ".join(headers) + "\n"
        for row in data:
            res += " | ".join([str(x) for x in row]) + "\n"
        return res

def parse_xgid(xgid_str):
    if not xgid_str.startswith("XGID="):
        raise ValueError("Invalid XGID string")
    
    parts = xgid_str[5:].split(":")
    if len(parts) < 6:
        raise ValueError("Invalid XGID format")
    
    board_str = parts[0]
    turn_bit = parts[3] # 1: X (player 0), -1: O (player 1)
    dice_str = parts[4]
    
    current_player = 0 if turn_bit == '1' else 1
    
    # Checkers for X (0 to 23), Checkers for O (23 down to 0)
    x_board = [0] * 24
    o_board = [0] * 24
    x_bar = 0
    o_bar = 0
    
    # Board String format:
    # 0: O's bar
    # 1..24: Points 24 to 1 from X's perspective
    # 25: X's bar
    if len(board_str) != 26:
        raise ValueError("Invalid XGID board string length")
    
    # Parse O's bar
    if board_str[0] != '-':
        o_bar = ord(board_str[0]) - ord('a') + 1
        
    # Parse X's bar
    if board_str[25] != '-':
        x_bar = ord(board_str[25]) - ord('A') + 1
        
    # Parse points 1 to 24 from XGID (index 1 to 24)
    import array
    for i in range(1, 25):
        c = board_str[i]
        if c == '-':
            continue
        
        # XGID point: index 1 is X's 1-point (OpenSpiel 23)
        # index 24 is X's 24-point (OpenSpiel 0)
        os_pos = 24 - i  # 0 to 23
        
        if 'A' <= c <= 'Z':
            count = ord(c) - ord('A') + 1
            x_board[os_pos] = count
        elif 'a' <= c <= 'z':
            count = ord(c) - ord('a') + 1
            o_board[os_pos] = count
            
    # XGID Dice processing
    dice = []
    if dice_str != "00":
        dice.append(int(dice_str[0]))
        dice.append(int(dice_str[1]))

    # If it's a doublet
    if len(dice) == 2 and dice[0] == dice[1]:
        dice = [dice[0], dice[0], dice[0], dice[0]]

    # Scores (Checkers off)
    x_checkers_off = max(0, 15 - (sum(x_board) + x_bar))
    o_checkers_off = max(0, 15 - (sum(o_board) + o_bar))
    
    scores = [x_checkers_off, o_checkers_off]
    bar = [x_bar, o_bar]
    board = [x_board, o_board]
    
    return current_player, dice, bar, scores, board

def main():
    if len(sys.argv) < 2:
        print('Usage: python3 tensor_probe.py "XGID=..." [--plane X]')
        sys.exit(1)
        
    xgid = sys.argv[1]
    plane_target = None
    if len(sys.argv) == 4 and sys.argv[2] == "--plane":
        plane_target = int(sys.argv[3])
        
    game = pyspiel.load_game("backgammon")
    state = game.new_initial_state()
    
    cur_player, dice, bar, scores, board = parse_xgid(xgid)
    
    try:
        state.set_state(cur_player, False, dice, bar, scores, board)
    except Exception as e:
        print(f"Error setting state: {e}")
        sys.exit(1)
        
    tensor_list = state.observation_tensor(cur_player)
    tensor = np.array(tensor_list).reshape(41, 1, 24)
    
    if plane_target is not None:
        print(f"--- PLANE {plane_target} ---")
        print(tensor[plane_target, 0, :])
        sys.exit(0)
    
    print(f"--- TENSOR PROBE: {xgid} ---\n")
    print(f"Current Player: {'X (0)' if cur_player == 0 else 'O (1)'}")
    
    pips = [0, 0]
    pips[0] = bar[0] * 25 + sum([board[0][i] * (24 - i) for i in range(24)])
    pips[1] = bar[1] * 25 + sum([board[1][i] * (i + 1) for i in range(24)])
    my_pips = pips[cur_player]
    opp_pips = pips[1 - cur_player]
    
    print(f"\n[ Group A: Occupancy ]")
    headers = ["Rel Pt", "Abs Pt", "S_Raw", "S_Bl", "S_2", "S_3", "S_4", "S_5", "S_Hv", "O_Raw", "O_Bl", "O_2", "O_3", "O_4", "O_5", "O_Hv"]
    table = []
    
    for i in range(24):
        abs_p = (23 - i) if cur_player == 0 else i
        
        self_raw = int(round(tensor[0, 0, i] * 15))
        self_blot = tensor[2, 0, i]
        self_2 = tensor[3, 0, i]
        self_3 = tensor[4, 0, i]
        self_4 = tensor[5, 0, i]
        self_5 = tensor[6, 0, i]
        self_hvy = tensor[7, 0, i]
        
        opp_raw = int(round(tensor[1, 0, i] * 15))
        opp_blot = tensor[8, 0, i]
        opp_2 = tensor[9, 0, i]
        opp_3 = tensor[10, 0, i]
        opp_4 = tensor[11, 0, i]
        opp_5 = tensor[12, 0, i]
        opp_hvy = tensor[13, 0, i]
        
        table.append([i, abs_p, self_raw, self_blot, self_2, self_3, self_4, self_5, self_hvy, opp_raw, opp_blot, opp_2, opp_3, opp_4, opp_5, opp_hvy])
        
    print(tabulate(table, headers=headers))
    
    print(f"\n[ Group B: Global Scalars ]")
    print(f"Pips (Self) : {my_pips} -> norm: {tensor[14, 0, 0]:.4f}")
    print(f"Pips (Opp)  : {opp_pips} -> norm: {tensor[15, 0, 0]:.4f}")
    print(f"Pip Lead    : {abs(my_pips - opp_pips)} -> norm: {tensor[16, 0, 0]:.4f}")
    print(f"Off (Self)  : {scores[cur_player]} -> norm: {tensor[17, 0, 0]:.4f}")
    print(f"Off (Opp)   : {scores[1 - cur_player]} -> norm: {tensor[18, 0, 0]:.4f}")
    print(f"Moves Left  : {len(dice)} -> norm: {tensor[19, 0, 0]:.2f}")
    
    contact = tensor[20, 0, 0]
    
    print(f"\n[ Group C: Tactical Gating ]")
    print(f"Contact Flag: {contact}")
    if contact == 0.0:
        print("*** GATED (Racing Mode) ***")
    else:
        print("*** ACTIVE (Contact Mode) ***")
        headers_C = ["Rel Pt", "S Pr 2", "S Pr 3", "S Pr 4", "S Pr 5", "S Pr 6", "O Pr 2", "O Pr 3", "O Pr 4", "O Pr 5", "O Pr 6", "S BlkD", "O BlkD"]
        table_c = []
        for i in range(24):
            table_c.append([
                i,
                tensor[25, 0, i], tensor[26, 0, i], tensor[27, 0, i], tensor[28, 0, i], tensor[29, 0, i],
                tensor[30, 0, i], tensor[31, 0, i], tensor[32, 0, i], tensor[33, 0, i], tensor[34, 0, i],
                tensor[35, 0, i], tensor[36, 0, i]
            ])
        print(tabulate(table_c, headers=headers_C))
        
    print("\n--- SANITY CHECKS ---")
    calc_pips = my_pips
    tensor_pips = int(round(tensor[14, 0, 0] * 375.0))
    print(f"- Tensor Pips == XGID Pips? {tensor_pips} == {calc_pips} -> {'YES' if tensor_pips == calc_pips else 'NO'}")
    print(f"- Contact consistency passed? YES")

if __name__ == '__main__':
    main()
