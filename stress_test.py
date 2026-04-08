import sys
sys.path.append("/home/jeremy/projects/open_spiel/build/python")
import pyspiel

game_std = pyspiel.load_game("backgammon", {"scoring_type": "full_scoring"})
print(f"[Stress 3] Standard Max Utility: {game_std.max_utility()}")

game_dmp = pyspiel.load_game("backgammon", {"scoring_type": "full_scoring", "dmp_only": True})
print(f"[Stress 3] DMP Max Utility: {game_dmp.max_utility()}")
