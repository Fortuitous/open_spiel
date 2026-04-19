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
import json
import traceback
import psutil
from google.cloud import storage
try:
    import wandb
except ImportError:
    wandb = None

from open_spiel.python.algorithms import mcts
from open_spiel.python.games.backgammon.expert_eyes_model import ExpertEyesNet
from open_spiel.python.games.backgammon.gnubg_auditor.universal_exporter import UniversalExporter

NUM_DISTINCT_ACTIONS = 913952
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RES_BLOCKS = 20
FILTERS = 256

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
                name = b.name.split("/")[-1]
                if "gen_" in name:
                    parts = name.split("_")
                    gen_id = int(parts[3].split(".")[0])
                    gen_numbers.append(gen_id)
            except (IndexError, ValueError):
                continue
        return max(gen_numbers) if gen_numbers else 0

def clean_dice_for_xg(dice_str):
    if "roll: " in dice_str:
        raw = dice_str.split("roll: ")[1].strip(")")
        sorted_dice = "".join(sorted(list(raw), reverse=True))
        return sorted_dice
    return dice_str

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
        tensor_input = torch.FloatTensor(obs_tensor).unsqueeze(0).to(DEVICE)
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
            
            # Form Sparse Pi Target
            non_zero_indices = []
            non_zero_probs = []
            valid_children_found = False
            
            total_explore = sum(child.explore_count for child in root.children)
            
            if total_explore > 0:
                for child in root.children:
                    if child.explore_count > 0:
                        non_zero_indices.append(child.action)
                        non_zero_probs.append(child.explore_count / total_explore)
                valid_children_found = True
                    
            if not valid_children_found:
                legal_actions = state.legal_actions()
                for a in legal_actions:
                    non_zero_indices.append(a)
                    non_zero_probs.append(1.0 / len(legal_actions))
                    
            pi_sparse = (np.array(non_zero_indices, dtype=np.int32), np.array(non_zero_probs, dtype=np.float32))
            game_history.append((player, obs_tensor, pi_sparse))
            
            children = root.children
            if children:
                sorted_children = sorted(children, key=lambda c: c.explore_count, reverse=True)
                action = sorted_children[0].action
            else:
                action = random.choice(state.legal_actions())
            
            if write_xg:
                move_str = format_moves_for_xg(state.action_to_string(player, action))
                if not move_records or move_records[-1]['player'] != player or move_records[-1]['dice'] != current_turn_dice:
                    move_records.append({
                        "player": player,
                        "dice": current_turn_dice,
                        "moves": [move_str]
                    })
                else:
                    move_records[-1]['moves'].append(move_str)
            
            state.apply_action(action)
            step_idx += 1

    returns = state.returns()
    formatted_data = []
    
    for (player_id, obs, pi) in game_history:
        z = returns[player_id]  
        formatted_data.append((obs, pi, z))
        
    if write_xg:
        winner_id = 0 if returns[0] > 0 else 1
        print_flush(f"DEBUG: Game Finished. Returns: {returns}, WinnerID: {winner_id}, File: {xg_filename}")
        UniversalExporter.write_to_file(xg_filename, move_records, winner_id=winner_id)
            
    return formatted_data

def signal_handler(sig, frame):
    print_flush(f"Caught signal {sig}, flushing logs and exiting safely...")
    sys.exit(0)

# Register signal handlers for Spot Instance preemption
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def train_iteration(gen_id, gcs, num_steps=200, lr=0.01, batch_size=128):
    print_flush(f"=== Starting Training Iteration (Iteration: {gen_id}) ===")
    print_flush(f"Using Hyperparams: LR={lr}, Batch={batch_size}, Steps={num_steps}")
    print_flush(f"Model Architecture: {RES_BLOCKS} Blocks, {FILTERS} Filters (Big Brain)")

    model = ExpertEyesNet(num_res_blocks=RES_BLOCKS, num_filters=FILTERS).to(DEVICE)
    
    # Download latest model from GCS
    latest_path = "checkpoints/latest.pt"
    local_latest = "latest_downloaded.pt"
    if gcs.download_file(latest_path, local_latest):
        model.load_state_dict(torch.load(local_latest, map_location=DEVICE, weights_only=False))
        print_flush(f"Fine-tuning from gs://{gcs.bucket.name}/{latest_path}")
    else:
        print_flush("No latest checkpoint found. Starting from SCRATCH.")

    # Streaming Ingestion
    print_flush("Streaming 1000 logs from GCS (logs/)...")
    blobs = gcs.list_recent_logs(prefix="logs/", limit=1000)
    print_flush(f"Found {len(blobs)} games in GCS (logs/)")
    buffer = ReplayBuffer(capacity=100000)
    
    for i, blob in enumerate(blobs):
        if (i + 1) % 50 == 0:
            print_flush(f"Ingesting game {i+1}/{len(blobs)}...")
        mem_percent = psutil.virtual_memory().percent
        if mem_percent > 85:
            print_flush(f"WARNING: Memory usage at {mem_percent}%. Stopping log ingestion.")
            break
            
        if blob.name.endswith(".pt_data"):
            local_temp = "current_game_ingest.pt_data"
            blob.download_to_filename(local_temp)
            
            try:
                game_data = torch.load(local_temp, weights_only=False)
                for d in game_data:
                    obs, pi, z = d
                    if not isinstance(pi, tuple):
                        mask = (pi > 1e-6)
                        indices = np.where(mask)[0]
                        probs = pi[mask]
                        pi_sparse = (indices.astype(np.int32), probs.astype(np.float32))
                        buffer.append((obs, pi_sparse, z))
                    else:
                        buffer.append(d)
            finally:
                if os.path.exists(local_temp):
                    os.remove(local_temp)

    print_flush(f"Total entries in Replay Buffer: {len(buffer)}")
    if len(buffer) == 0:
        print_flush("ERROR: No data found for training!")
        return

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    
    actual_batch_size = min(batch_size, len(buffer))
    
    for step in range(num_steps):
        batch = buffer.sample(actual_batch_size)
        obs_batch = torch.FloatTensor(np.array([x[0] for x in batch])).to(DEVICE)
        z_batch = torch.FloatTensor(np.array([x[2] for x in batch])).to(DEVICE).unsqueeze(1)
        
        pi_batch = torch.zeros((len(batch), NUM_DISTINCT_ACTIONS), dtype=torch.float32).to(DEVICE)
        for i, (_, pi_data, _) in enumerate(batch):
            if isinstance(pi_data, tuple) and len(pi_data) == 2:
                indices, probs = pi_data
                pi_batch[i, indices] = torch.FloatTensor(probs).to(DEVICE)
            else:
                pi_batch[i] = torch.FloatTensor(pi_data).to(DEVICE)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            p_logits, v_pred = model(obs_batch)
            v_loss = F.mse_loss(v_pred, z_batch)
            log_p = F.log_softmax(p_logits, dim=1)
            p_loss = -torch.mean(torch.sum(pi_batch * log_p, dim=1))
            loss = v_loss + p_loss
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if (step + 1) % 50 == 0:
            print_flush(f"  [Step {step+1:04}/{num_steps}] V_Loss: {v_loss.item():.4f} | P_Loss: {p_loss.item():.4f}")
            if wandb and wandb.run:
                wandb.log({
                    "train/step": step + 1,
                    "train/v_loss": v_loss.item(),
                    "train/p_loss": p_loss.item(),
                    "train/total_loss": loss.item(),
                })
        
        if (step + 1) % 1000 == 0:
            periodic_ckpt = f"expert_eyes_gen_{gen_id}_step_{step+1}.pt"
            torch.save(model.state_dict(), periodic_ckpt)
            gcs.upload_file(periodic_ckpt, f"checkpoints/periodic/{periodic_ckpt}")
            os.remove(periodic_ckpt)

    gen_ckpt = f"expert_eyes_gen_{gen_id}.pt"
    torch.save(model.state_dict(), gen_ckpt)
    gcs.upload_file(gen_ckpt, f"checkpoints/{gen_ckpt}")
    gcs.upload_file(gen_ckpt, "checkpoints/latest.pt")
    print_flush(f"Iteration {gen_id} complete.")

def play_mode(gcs, num_games=10, max_sims=400, job_id=None):
    print_flush(f"=== Starting Play Mode ({num_games} Games, {max_sims} Simulations) ===")
    game = pyspiel.load_game("backgammon(dmp_only=True)")
    model = ExpertEyesNet(num_res_blocks=20, num_filters=256).to(DEVICE)
    
    local_latest = "latest_worker.pt"
    if gcs.download_file("checkpoints/latest.pt", local_latest):
        model.load_state_dict(torch.load(local_latest, map_location=DEVICE, weights_only=False))
        print_flush(f"Loaded latest checkpoint from GCS onto {DEVICE}.")
    else:
        print_flush("ERROR: No latest.pt found on GCS!")
        return

    model.eval()
    evaluator = ExpertEyesEvaluator(model)
    
    job_id = job_id or os.environ.get("CLOUD_ML_JOB_ID", str(int(time.time())))
    worker_id = get_vertex_worker_id()
    print_flush(f"Assigned Identity: Worker {worker_id} (Job: {job_id})")
    
    for idx in range(1, num_games + 1):
        try:
            xg_filename = f"game_{job_id}_w{worker_id}_{idx}_xg.txt"
            data_filename = f"game_{job_id}_w{worker_id}_{idx}.pt_data"
            
            if gcs.exists(f"logs/{xg_filename}"):
                print_flush(f"  Game {idx}/{num_games} already exists. Skipping.")
                continue
                
            print_flush(f"  [START] Playing Game {idx}/{num_games}...")
            game_data = play_training_game(game, model, evaluator, write_xg=True, xg_filename=xg_filename, max_sims=max_sims)
            torch.save(game_data, data_filename)
            gcs.upload_file(xg_filename, f"logs/{xg_filename}")
            gcs.upload_file(data_filename, f"logs/{data_filename}")
            os.remove(xg_filename)
            os.remove(data_filename)
            print_flush(f"  [COMPLETE] Game {idx}/{num_games} uploaded.")
        except Exception as e:
            print_flush(f"CRITICAL ERROR in Game {idx}: {str(e)}")
            traceback.print_exc()
            continue 
        
    print_flush("Play session complete.")

def init_scratch(gcs):
    print_flush("=== Initializing Clean Slate Model (20 Blocks, 256 Filters) ===")
    model = ExpertEyesNet(num_res_blocks=RES_BLOCKS, num_filters=FILTERS).to(DEVICE)
    ckpt_name = "expert_eyes_gen_0.pt"
    torch.save(model.state_dict(), ckpt_name)
    gcs.upload_file(ckpt_name, f"checkpoints/{ckpt_name}")
    gcs.upload_file(ckpt_name, "checkpoints/latest.pt")
    os.remove(ckpt_name)
    print_flush("Initialization complete.")

def get_vertex_worker_id():
    cluster_spec_str = os.environ.get("CLUSTER_SPEC")
    if cluster_spec_str:
        try:
            spec = json.loads(cluster_spec_str)
            task = spec.get("task", {})
            pool_name = task.get("type", "workerpool")
            index = task.get("index", 0)
            return f"{pool_name}_{index}"
        except Exception:
            pass
    return os.environ.get("CLOUD_ML_TASK_ID", os.environ.get("AIP_TASK_INDEX", "0"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["play", "train", "init_scratch"], default="play")
    parser.add_argument("--bucket_name", type=str, default="expert-eyes-training-742")
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--job_id", type=str, default=None)
    args = parser.parse_args()
    
    if args.wandb and wandb and os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        gcs = GCSHelper(args.bucket_name)
        current_gen = gcs.get_latest_checkpoint_info()
        run_name = f"gen_{current_gen}_{args.mode}_{int(time.time())}"
        job_id = os.environ.get("CLOUD_ML_JOB_ID", str(int(time.time())))
        wandb.init(project="expert-eyes-backgammon", name=run_name, id=job_id, resume="allow")

    gcs = GCSHelper(args.bucket_name)
    if args.mode == "init_scratch":
        init_scratch(gcs)
    elif args.mode == "play":
        play_mode(gcs, num_games=args.num_games, max_sims=args.sims, job_id=args.job_id)
    elif args.mode == "train":
        latest_gen = gcs.get_latest_checkpoint_info()
        train_iteration(latest_gen + 1, gcs, num_steps=args.steps, lr=args.lr, batch_size=args.batch_size)
