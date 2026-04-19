import os
import re

class UniversalExporter:
    """
    State-of-the-art Backgammon Game Exporter.
    Produces "Flat Snowie" format which is highly compatible with GNUbg and XG.
    """
    
    @staticmethod
    def get_snowie_header(player1_name="Player 1", player2_name="Player 2", match_length=1, game_num=1):
        return (
            "; [Site \"Expert Eyes Trainer v1.0\"]\n"
            "; [Variation \"Backgammon\"]\n\n"
            f"{match_length} point match\n\n\n"
            f" Game {game_num}\n"
            f" {player1_name} : 0                         {player2_name} : 0\n"
        )

    @staticmethod
    def format_game(move_records, player1="Player 1", player2="Player 2", winner_id=0):
        """
        Formats the moves into a Snowie-compatible text format.
        Ensures Player 0 is always in the Left column and Player 1 is always in the Right.
        """
        output = UniversalExporter.get_snowie_header(player1, player2)
        
        # Group records by turn (a turn is a sequence of X move followed by O move)
        turns = []
        current_turn = {"p0": None, "p1": None}
        
        for rec in move_records:
            p = rec['player']
            if p == 0:
                if current_turn["p0"] is not None:
                    # Start new turn
                    turns.append(current_turn)
                    current_turn = {"p0": rec, "p1": None}
                else:
                    current_turn["p0"] = rec
            else:
                if current_turn["p1"] is not None:
                    # Start new turn
                    turns.append(current_turn)
                    current_turn = {"p0": None, "p1": rec}
                else:
                    current_turn["p1"] = rec
        
        if current_turn["p0"] or current_turn["p1"]:
            turns.append(current_turn)

        for i, turn in enumerate(turns):
            move_num = i + 1
            p0 = turn["p0"]
            p1 = turn["p1"]
            
            p0_dice = p0['dice'] if p0 else ""
            p0_moves = " ".join(p0['moves']) if p0 else ""
            
            p1_dice = p1['dice'] if p1 else ""
            p1_moves = " ".join(p1['moves']) if p1 else ""
            
            # Format doubles
            if p0 and len(set(p0['dice'])) == 1:
                for m in set(p0['moves']):
                    p0_moves = p0_moves.replace(f"{m} {m} {m} {m}", f"{m}(4)")
                    p0_moves = p0_moves.replace(f"{m} {m} {m}", f"{m}(3)")
                    p0_moves = p0_moves.replace(f"{m} {m}", f"{m}(2)")
            
            if p1 and len(set(p1['dice'])) == 1:
                for m in set(p1['moves']):
                    p1_moves = p1_moves.replace(f"{m} {m} {m} {m}", f"{m}(4)")
                    p1_moves = p1_moves.replace(f"{m} {m} {m}", f"{m}(3)")
                    p1_moves = p1_moves.replace(f"{m} {m}", f"{m}(2)")
            
            line = f"  {move_num:2}) {p0_dice}: {p0_moves:<28} {p1_dice}: {p1_moves}"
            output += line + "\n"
            
        # Winner alignment: Player 0 is Left, Player 1 is Right.
        if winner_id == 0:
            output += "  Wins 1 point\r\n"
        else:
            output += f"{' ':>40}Wins 1 point\r\n"
        return output

    @staticmethod
    def write_to_file(filename, move_records, player1="Player 1", player2="Player 2", winner_id=0):
        content = UniversalExporter.format_game(move_records, player1, player2, winner_id)
        with open(filename, "w") as f:
            f.write(content)
        return filename
