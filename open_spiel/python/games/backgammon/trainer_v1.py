import sys
sys.path.append("/home/jeremy/projects/open_spiel")
sys.path.append("/home/jeremy/projects/open_spiel/build/python")

import torch
import torch.nn.functional as F
import torch.optim as optim
import pyspiel
import numpy as np
from collections import deque
import random
import os
import argparse
import signal
import time
import shutil
import glob
from google.cloud import storage
try:
    import wandb
except ImportError:
    wandb = None

from open_spiel.python.algorithms import mcts
from open_spiel.python.games.backgammon.expert_eyes_model import ExpertEyesNet

NUM_DISTINCT_ACTIONS = 913952

# Ensure prints are flushed immediately for cloud logging
def print_flush(msg):
    print(msg, flush=True)

class GCSHelper:
    def __init__(self, bucket_name):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, local_path, remote_path):
        blob = self.bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        print_flush(f"Uploaded {local_path} to gs://{self.bucket.name}/{remote_path}")

    def exists(self, remote_path):
        return self.bucket.blob(remote_path).exists()

    def download_file(self, remote_path, local_path):
        blob = self.bucket.blob(remote_path)
        if not blob.exists():
            return False
        blob.download_to_filename(local_path)
        print_flush(f"Downloaded gs://{self.bucket.name}/{remote_path} to {local_path}")
        return True

    def list_recent_logs(self, prefix="logs/", limit=1000):
        blobs = list(self.client.list_blobs(self.bucket, prefix=prefix))
        # Sort by updated time, newest first
        blobs.sort(key=lambda x: x.updated, reverse=True)
        return blobs[:limit]

    def cleanup_old_logs(self, prefix="logs/", keep=5000):
        blobs = list(self.client.list_blobs(self.bucket, prefix=prefix))
        if len(blobs) <= keep:
            return
        blobs.sort(key=lambda x: x.updated, reverse=True)
        to_delete = blobs[keep:]
        print_flush(f"Cleaning up {len(to_delete)} old logs from GCS...")
        with self.client.batch():
            for b in to_delete:
                b.delete()

    def get_latest_checkpoint_info(self):
        blobs = list(self.client.list_blobs(self.bucket, prefix="checkpoints/expert_eyes_gen_"))
        if not blobs:
            return 0
        gen_numbers = []
        for b in blobs:
            try:
                # expert_eyes_gen_N.pt
                parts = b.name.split("/")[-1].split("_")
                if len(parts) >= 4:
                    gen_id = int(parts[3].split(".")[0])
                    gen_numbers.append(gen_id)
            except (IndexError, ValueError):
                continue
        return max(gen_numbers) if gen_numbers else 0

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

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def append(self, item):
        self.buffer.append(item)
        
    def __len__(self):
        return len(self.buffer)
        
    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

def play_training_game(game, model, evaluator, write_xg=False, xg_filename="smoke_test_xg.txt", max_sims=40):
    mcts_bot = mcts.MCTSBot(
        game,
        uct_c=1.1,
        max_simulations=max_sims,
        evaluator=evaluator,
        solve=True
    )
    
    state = game.new_initial_state()
    move_records = [] 
    current_turn_dice = ""
    
    game_history = [] 

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
            obs_tensor = state.observation_tensor()
            
            root = mcts_bot.mcts_search(state)
            
            # Form Pi Target
            pi = np.zeros(NUM_DISTINCT_ACTIONS, dtype=np.float32)
            valid_children_found = False
            for child in root.children:
                if child.explore_count > 0:
                    pi[child.action] = child.explore_count
                    valid_children_found = True
                    
            if valid_children_found:
                pi /= np.sum(pi)
            else:
                legal_actions = state.legal_actions()
                for a in legal_actions:
                    pi[a] = 1.0 / len(legal_actions)
                    
            game_history.append((player, obs_tensor, pi))
            
            children = root.children
            sorted_children = sorted(children, key=lambda c: c.explore_count, reverse=True)
            action = sorted_children[0].action
            
            if write_xg:
                move_str = format_moves_for_xg(state.action_to_string(player, action))
                move_records.append({
                    "player": player,
                    "dice": current_turn_dice,
                    "move": move_str
                })
            
            state.apply_action(action)
            step_idx += 1

    returns = state.returns()
    formatted_data = []
    
    for (player_id, obs, pi) in game_history:
        z = returns[player_id]  
        formatted_data.append((obs, pi, z))
        
    if write_xg:
        with open(xg_filename, "w") as f:
            f.write("; [Site \"Expert Eyes Trainer v1.0\"]\n")
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
            
    return formatted_data

def signal_handler(sig, frame):
    print_flush(f"Caught signal {sig}, flushing logs and exiting safely...")
    sys.exit(0)

# Register signal handlers for Spot Instance preemption
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def train_iteration(gen_id, gcs, num_steps=200):
    print_flush(f"=== Starting Training Iteration (New Gen: {gen_id}) ===")
    
    tmp_dir = "./tmp_training"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    # Scale: 20 blocks, 256 filters
    model = ExpertEyesNet(num_res_blocks=20, num_filters=256)
    
    # Download latest model from GCS
    latest_path = "checkpoints/latest.pt"
    local_latest = "latest_downloaded.pt"
    if gcs.download_file(latest_path, local_latest):
        model.load_state_dict(torch.load(local_latest, weights_only=False))
        print_flush(f"Fine-tuning from gs://{gcs.bucket.name}/{latest_path}")
    else:
        print_flush("No latest checkpoint found. Starting from SCRATCH (Ensure this is intended).")
    
    # Download 1000 most recent logs
    print_flush("Downloading sliding window of 1000 logs from GCS...")
    blobs = gcs.list_recent_logs(limit=1000)
    buffer = ReplayBuffer(capacity=100000)
    
    for blob in blobs:
        local_log = os.path.join(tmp_dir, blob.name.split("/")[-1])
        blob.download_to_filename(local_log)
        # Parse log (This assumes we can reconstruct data from XG logs or we save raw data)
        # For simplicity, we assume the trainer saves .pt data or we need a raw format.
        # [Wait] The original trainer returned list of (obs, pi, z). 
        # We need to save that to GCS as well. I'll update the 'play' mode to save .pt data.
        if local_log.endswith(".pt_data"):
            game_data = torch.load(local_log, weights_only=False)
            for d in game_data:
                buffer.append(d)

    print_flush(f"Total entries in Replay Buffer: {len(buffer)}")
    if len(buffer) == 0:
        print_flush("ERROR: No data found for training!")
        return

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    model.train()
    batch_size = 128
    actual_batch_size = min(batch_size, len(buffer))
    
    for step in range(num_steps):
        batch = buffer.sample(actual_batch_size)
        obs_batch = torch.FloatTensor(np.array([x[0] for x in batch]))
        pi_batch = torch.FloatTensor(np.array([x[1] for x in batch]))
        z_batch = torch.FloatTensor(np.array([x[2] for x in batch])).unsqueeze(1)
        
        optimizer.zero_grad()
        p_logits, v_pred = model(obs_batch)
        v_loss = F.mse_loss(v_pred, z_batch)
        log_p = F.log_softmax(p_logits, dim=1)
        p_loss = -torch.mean(torch.sum(pi_batch * log_p, dim=1))
        
        loss = v_loss + p_loss
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 50 == 0:
            print_flush(f"  [Step {step+1:03}] V_Loss: {v_loss.item():.4f} | P_Loss: {p_loss.item():.4f}")
            if wandb and wandb.run:
                wandb.log({
                    "train/step": step + 1,
                    "train/v_loss": v_loss.item(),
                    "train/p_loss": p_loss.item(),
                    "train/total_loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                })

    # Save and upload
    gen_ckpt = f"expert_eyes_gen_{gen_id}.pt"
    torch.save(model.state_dict(), gen_ckpt)
    gcs.upload_file(gen_ckpt, f"checkpoints/{gen_ckpt}")
    gcs.upload_file(gen_ckpt, "checkpoints/latest.pt")
    
    # Cleanup
    gcs.cleanup_old_logs(keep=5000)
    shutil.rmtree(tmp_dir)
    print_flush(f"Iteration {gen_id} complete.")

def play_mode(gcs, num_games=10, max_sims=400):
    print_flush(f"=== Starting Play Mode ({num_games} Games, {max_sims} Simulations) ===")
    game = pyspiel.load_game("backgammon(dmp_only=True)")
    
    # Scale: 20 blocks, 256 filters
    model = ExpertEyesNet(num_res_blocks=20, num_filters=256)
    
    local_latest = "latest_worker.pt"
    if gcs.download_file("checkpoints/latest.pt", local_latest):
        model.load_state_dict(torch.load(local_latest, weights_only=False))
        print_flush("Loaded latest checkpoint from GCS.")
    else:
        print_flush("ERROR: No latest.pt found on GCS! Run with --mode init_scratch first.")
        return

    model.eval()
    evaluator = ExpertEyesEvaluator(model)
    
    # Resilience IDs: Use command line arg -> Environment variable -> Default
    job_id = args.job_id or os.environ.get("CLOUD_ML_JOB_ID", str(int(time.time())))
    worker_id = args.worker_id or os.environ.get("CLOUD_ML_TASK_ID", "0")
    
    for g in range(num_games):
        # Pattern: game_{job_id}_w{worker_id}_{index}
        xg_filename = f"game_{job_id}_w{worker_id}_{g+1}_xg.txt"
        data_filename = f"game_{job_id}_w{worker_id}_{g+1}.pt_data"
        
        # Check if this specific game already exists in GCS
        if gcs.exists(f"logs/{xg_filename}"):
            print_flush(f"  Game {g+1}/{num_games} (Worker {worker_id}) already exists in GCS. Skipping.")
            continue
            
        print_flush(f"  Playing Game {g+1}/{num_games} (Worker {worker_id})...")
        game_data = play_training_game(game, model, evaluator, write_xg=True, xg_filename=xg_filename, max_sims=max_sims)
        
        # Save raw data for trainer
        torch.save(game_data, data_filename)
        
        # Upload both to GCS
        gcs.upload_file(xg_filename, f"logs/{xg_filename}")
        gcs.upload_file(data_filename, f"logs/{data_filename}")
        
        # Local Cleanup
        os.remove(xg_filename)
        os.remove(data_filename)

        if wandb and wandb.run:
            # Game stats
            returns = game_data[0][2] if game_data else 0 # z from first entry (terminal return)
            wandb.log({
                "play/game_length": len(game_data),
                "play/game_result": returns, # +1 or -1
                "play/generation": gcs.get_latest_checkpoint_info()
            })
        
    print_flush("Play session complete.")

def init_scratch(gcs):
    print_flush("=== Initializing Clean Slate Model (20 Blocks, 256 Filters) ===")
    model = ExpertEyesNet(num_res_blocks=20, num_filters=256)
    ckpt_name = "expert_eyes_gen_0.pt"
    torch.save(model.state_dict(), ckpt_name)
    gcs.upload_file(ckpt_name, f"checkpoints/{ckpt_name}")
    gcs.upload_file(ckpt_name, "checkpoints/latest.pt")
    os.remove(ckpt_name)
    print_flush("Initialization complete. Model gs://{gcs.bucket.name}/checkpoints/latest.pt is ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["play", "train", "init_scratch"], default="play")
    parser.add_argument("--bucket_name", type=str, default="expert-eyes-training-742")
    parser.add_argument("--num_games", type=int, default=10, help="Number of games to play in 'play' mode")
    parser.add_argument("--sims", type=int, default=400, help="MCTS simulations per move")
    parser.add_argument("--steps", type=int, default=200, help="Optimization steps in 'train' mode")
    parser.add_argument("--wandb", action="store_true", help="Explicit toggle for W&B telemetry")
    parser.add_argument("--job_id", type=str, default=None, help="Explicit Job ID for resumption testing")
    parser.add_argument("--worker_id", type=str, default=None, help="Explicit Worker ID for resumption testing")
    args = parser.parse_args()
    
    # W&B Logic
    if args.wandb and wandb and os.environ.get("WANDB_API_KEY"):
        print_flush("W&B API Key detected. Initializing telemetry...")
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        
        # Determine Gen for run naming
        gcs = GCSHelper(args.bucket_name)
        current_gen = gcs.get_latest_checkpoint_info()
        run_name = f"gen_{current_gen}_{args.mode}_{int(time.time())}"
        
        # Auto-config architecture
        temp_model = ExpertEyesNet(num_res_blocks=20, num_filters=256)
        config = {
            "mode": args.mode,
            "resnet_blocks": 20,
            "filters": 256,
            "mcts_sims": args.sims,
            "learning_rate": 0.01,
            "batch_size": 128,
            "architecture": str(temp_model)
        }
        
        job_id = os.environ.get("CLOUD_ML_JOB_ID", str(int(time.time())))
        wandb.init(
            project="expert-eyes-backgammon",
            name=run_name,
            id=job_id,
            resume="allow",
            config=config
        )
    else:
        print_flush("W&B API Key NOT found. Telemetry disabled.")

    gcs = GCSHelper(args.bucket_name)
    
    if args.mode == "init_scratch":
        init_scratch(gcs)
    elif args.mode == "play":
        play_mode(gcs, num_games=args.num_games, max_sims=args.sims)
    elif args.mode == "train":
        latest_gen = gcs.get_latest_checkpoint_info()
        train_iteration(latest_gen + 1, gcs, num_steps=args.steps)
