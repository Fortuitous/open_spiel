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
    def format_game(move_records, player1="Player 1", player2="Player 2"):
        """
        Formats a full game into the exact 'Good' Snowie Text string.
        """
        output = UniversalExporter.get_snowie_header(player1, player2)
        
        for i in range(0, len(move_records), 2):
            move_num = (i // 2) + 1
            p1 = move_records[i]
            p1_move = " ".join(p1['moves'])
            if len(set(p1['dice'])) == 1: # It's a double
                for move in set(p1['moves']):
                    p1_move = p1_move.replace(f"{move} {move} {move} {move}", f"{move}(4)")
                    p1_move = p1_move.replace(f"{move} {move} {move}", f"{move}(3)")
                    p1_move = p1_move.replace(f"{move} {move}", f"{move}(2)")
            
            p2_move = ""
            if i + 1 < len(move_records):
                p2 = move_records[i+1]
                p2_move = " ".join(p2['moves'])
                if len(set(p2['dice'])) == 1: # It's a double
                    for move in set(p2['moves']):
                        p2_move = p2_move.replace(f"{move} {move} {move} {move}", f"{move}(4)")
                        p2_move = p2_move.replace(f"{move} {move} {move}", f"{move}(3)")
                        p2_move = p2_move.replace(f"{move} {move}", f"{move}(2)")
            
            line = f"  {move_num:2}) {p1['dice']}: {p1_move:<28} {move_records[i+1]['dice'] if i+1 < len(move_records) else ''}: {p2_move}"
            output += line + "\n"
            
        output += "  Wins 1 point\r\n"
        return output

    @staticmethod
    def write_to_file(filename, move_records, player1="Player 1", player2="Player 2"):
        content = UniversalExporter.format_game(move_records, player1, player2)
        with open(filename, "w") as f:
            f.write(content)
        return filename
