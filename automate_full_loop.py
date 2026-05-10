import subprocess
import time
import json
import os
import re

PROJECT_ID = "expert-eyes-training-742"
REGION = "us-central1"

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"Error: {proc.stderr}", flush=True)
    return proc.stdout, proc.returncode

def submit_job(display_name, config_file):
    print(f"Submitting job: {display_name} with config {config_file}", flush=True)
    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        f"--project={PROJECT_ID}",
        f"--region={REGION}",
        f"--display-name={display_name}",
        f"--config={config_file}"
    ]
    stdout, rc = run_cmd(cmd)

    match = re.search(r'projects/\d+/locations/[^/]+/customJobs/\d+', stdout)
    if not match:
        cmd_stderr = ["gcloud", "ai", "custom-jobs", "create", f"--project={PROJECT_ID}", f"--region={REGION}", f"--display-name={display_name}", f"--config={config_file}"]
        proc = subprocess.run(cmd_stderr, capture_output=True, text=True)
        match = re.search(r'projects/\d+/locations/[^/]+/customJobs/\d+', proc.stderr)

    if match:
        job_name = match.group(0)
        print(f"Submitted successfully. Job Name: {job_name}", flush=True)
        return job_name
    print("Could not parse job name.", flush=True)
    return None

def wait_for_job(job_name, poll_interval=3600):
    print(f"Polling job {job_name} every {poll_interval} seconds...", flush=True)
    while True:
        cmd = [
            "gcloud", "ai", "custom-jobs", "describe", job_name,
            "--format=json"
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"Error describing job: {proc.stderr}", flush=True)
            time.sleep(poll_interval)
            continue

        job_info = json.loads(proc.stdout)
        state = job_info.get("state", "UNKNOWN")
        print(f"Current state: {state}", flush=True)

        if state == "JOB_STATE_SUCCEEDED":
            print(f"Job {job_name} succeeded.", flush=True)
            return True
        elif state in ["JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"]:
            print(f"Job {job_name} failed with state {state}.", flush=True)
            return False

        time.sleep(poll_interval)

def execute_pipeline(gen_name):
    print(f"\n--- Executing Pipeline for Gen {gen_name} ---", flush=True)

    collection_job = None
    if gen_name == "2.1":
        cmd = ["gcloud", "ai", "custom-jobs", "list", f"--region={REGION}", "--limit=1", "--format=json"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            jobs = json.loads(proc.stdout)
            if jobs and jobs[0].get("displayName") == "phase-1-gen2-mass-collection":
                collection_job = jobs[0].get("name")
                print(f"Found existing collection job: {collection_job}", flush=True)
        except Exception as e:
            print(f"Failed to fetch jobs: {e}", flush=True)

    # 1. Collection
    if not collection_job:
        collection_job = submit_job(f"phase-1-gen{gen_name}-mass-collection", "mass_collection_gen2.yaml")
        if not collection_job:
            return False

    success = wait_for_job(collection_job, poll_interval=3600)
    if not success: return False

    # 2. Training
    train_job = submit_job(f"phase-1-gen{gen_name}-training", "mass_collection_gen2_train.yaml")
    if not train_job: return False
    success = wait_for_job(train_job, poll_interval=1800)
    if not success: return False

    # 3. Eval Collect - Greedy & MCTS
    greedy_job = submit_job(f"diag-gen{gen_name}-no-mcts", "diag_no_mcts.yaml")
    mcts_job = submit_job(f"diag-gen{gen_name}-mcts", "diag_mcts_400.yaml")

    if greedy_job: wait_for_job(greedy_job, poll_interval=600)
    if mcts_job: wait_for_job(mcts_job, poll_interval=1800)

    # 4. Audit & Publish
    print(f"Running audit script for gen {gen_name}...", flush=True)
    run_cmd(["python3", "open_spiel/python/games/backgammon/gnubg_auditor/generational_auditor.py", gen_name])

    print(f"Pipeline for Gen {gen_name} completed successfully.", flush=True)
    return True

if __name__ == "__main__":
    print("Starting automated loop...", flush=True)
    execute_pipeline("2.1")
