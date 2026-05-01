# JulesWeb Automation Handoff Manifesto: Gen 2.0 Backgammon

Welcome, JulesWeb. This document outlines the current state of the Gen 2.0 Backgammon engine repository and provides explicit instructions for executing the automated training loop.

> [!CAUTION]
> **ABSOLUTE ZERO RESTART MANDATE**
> The C++ Observation Tensor was just overhauled from v12.1 (1152 elements) to v12.3 (1220 elements). **The `latest.pt` file currently in GCS is INCOMPATIBLE with the new engine.**
> Your very first task for Gen 2.0 must be an "Absolute Zero" restart: initialize a fresh, randomly weighted `ExpertEyesNet` using the 1220 schema, upload it as the baseline `latest.pt`, and begin data collection from there.

## 1. The 'Core Four' Script Map

### 1. `trainer_v1.py`
The primary workhorse for Mass Collection and Training.
- **Mass Collection**: `python3 trainer_v1.py --mode play --num_games 1000 --sims 400 --bucket_name expert-eyes-training-742`
- **Model Training**: `python3 trainer_v1.py --mode train --steps 1000 --batch_size 128 --bucket_name expert-eyes-training-742`

### 2. `gnubg_auditor/generational_auditor.py`
The brand new unified evaluation script.
- **Command**: `python3 gnubg_auditor/generational_auditor.py [GEN_NAME]` (e.g., `python3 gnubg_auditor/generational_auditor.py 2.0`)
- Automatically downloads diagnostic Greedy and MCTS games from GCS, runs GNUbg 2-ply evaluation, generates `audit_run_[GEN_NAME].txt`, appends to `master_audit_summary.txt`, and uploads both to GCS (`gs://expert-eyes-training-742/summaries/`).

### 3. `tensor_audit.py`
Used for spatial sanity-checking and local debugging.
- **Command**: `python3 tensor_audit.py "XGID=..."`

### 4. Vertex AI Orchestration YAMLs
- `mass_collection_gen2.yaml` / `mass_collection_gen2_train.yaml`
- `diag_no_mcts.yaml` / `diag_mcts_400.yaml` (Use these to trigger the specific 10-game diagnostic evaluation collections).

## 2. GCS Infrastructure & Schema

**Bucket**: `gs://expert-eyes-training-742`
- `/checkpoints`: Contains `latest.pt` and `expert_eyes_gen_[X].pt`.
- `/logs`: Contains self-play `.pt_data` and `_xg.txt` trajectory logs.
- `/summaries`: Contains `master_audit_summary.txt` and `audit_run_[X].txt` files.
- `/gold_standard_audits`: Protected benchmark directory. **Do not overwrite or mutate files in this directory during training loops.**

## 3. Operational 'Tribal Knowledge'

- **OOM Thresholds**: The 50-plane ResNet architecture is highly memory-intensive. When training on an NVIDIA Tesla T4, keep `--batch_size` at `128` to prevent Out-Of-Memory (OOM) fatal errors.
- **GCS Handshake**: `trainer_v1.py` implements explicit `time.sleep(5)` cooldowns and batch uploading logic. Do not attempt to strip these out; they are necessary to prevent GCS API rate limiting during parallel mass ingestion.

## 4. The Automated Gen 2.0 Loop

To ensure a perfectly clean baseline, JulesWeb must execute this in two phases:

### Phase 1: The Gen 2.0 Baseline Initialization
*(Do this exactly once to establish the randomly weighted benchmark)*
1. **Initialize Weights:** Run `trainer_v1.py` with `--mode init_scratch` (or similar) to generate and upload the random 1220-schema `latest.pt`.
2. **Eval Collect:** Trigger `diag_no_mcts.yaml` and `diag_mcts_400.yaml`. Ensure these output to GCS with the prefix `logs/diag_gen2.0_`.
3. **Audit:** Run `python3 gnubg_auditor/generational_auditor.py 2.0` to generate the absolute baseline summary logs.
4. **Publish:** Read `master_audit_summary.txt` from GCS and update the external Google Doc.

### Phase 2: The Infinite 2.x Training Loop
*(Start tracking at `GEN_NAME = 2.1` and loop infinitely)*
1. **Mass Collect:** Trigger `mass_collection_gen2.yaml` to self-play 1000 MCTS games.
2. **Train:** Trigger `mass_collection_gen2_train.yaml` to update weights based on those 1000 games.
3. **Eval Collect:** Trigger `diag_no_mcts.yaml` and `diag_mcts_400.yaml`. Ensure these output to GCS with the prefix `logs/diag_gen[GEN_NAME]_`.
4. **Audit:** Run `python3 gnubg_auditor/generational_auditor.py [GEN_NAME]` to append to the master summary.
5. **Publish & Repeat:** Read the updated `master_audit_summary.txt` from GCS, update the external Google Doc, increment the `GEN_NAME` tracker (e.g., to 2.2), and restart from Step 1.

## 5. Annotated Directory

- `expert_eyes_model.py`: PyTorch Neural Network architecture (ResNet Tower, 1220 input dimensions).
- `self_play_v1.py` / `self_play_v2.py`: Legacy iteration scripts for MCTS self-play.
- `trainer_v1.py`: Master CLI tool for driving GCS-integrated data collection and training.
- `tensor_audit.py`: Diagnostic tool for visualizing the 1220-element observation tensor from XGIDs.
- `gnubg_auditor/generational_auditor.py`: Consolidates and analyzes Gen 2.x Greedy/MCTS diagnostic outputs.
- `gnubg_auditor/batch_auditor.py`: Legacy auditor for bulk GCS game evaluations.
- `gnubg_auditor/sampled_auditor.py`: Legacy auditor for slicing time-based batches out of mass-collection runs.
- `gnubg_auditor/dna_integrity_test.py`: Sanity-checks valid move generation against GNUbg benchmarks.
- `scripts/`: Contains protected scratch testing files (`action_test.py`, `random_game.py`, `play_and_audit.py`).
