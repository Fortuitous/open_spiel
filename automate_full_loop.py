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

def execute_phase2():
    print("\n--- Executing Phase 2: Cycle 2.2 ---", flush=True)

    # 1. Collection 2.2 (Already submitted by a previous run, using the running Job ID 1006421733508382720)
    collection_job = "projects/947210424180/locations/us-central1/customJobs/1006421733508382720"
    success = wait_for_job(collection_job, poll_interval=3600)
    if not success: return

    # 2. Training 2.2
    train_job = submit_job("phase-2-gen2.2-training", "mass_collection_gen2_train.yaml")
    if not train_job: return
    success = wait_for_job(train_job, poll_interval=1800)
    if not success: return

    # 3. Eval Collect 2.3 - Greedy & Mcripts
    greedy_job_2 = submit_job("diag-gen2.3-no-mcts", "diag_no_mcts.yaml")
    mcts_job_2 = submit_job("diag-gen2.3-mcts", "diag_mcts_400.yaml")

    if greedy_job_2: wait_for_job(greedy_job_2, poll_interval=600)
    if mcts_job_2: wait_for_job(mcts_job_2, poll_interval=600)

    print("Curating 2.3 transcripts...", flush=True)
    curate_cmd = [
        "python3", "-c",
        "from google.cloud import storage; "
        "client = storage.Client(); "
        "bucket = client.bucket('expert-eyes-training-742'); "
        "blobs = bucket.list_blobs(prefix='logs/'); "
        "[(bucket.copy_blob(b, bucket, b.name.replace('logs/', 'audits/gen_2.3_transcripts/', 1))) for b in blobs if 'diag_gen1.2' in b.name and b.name.endswith('_xg.txt')]"
    ]
    run_cmd(curate_cmd)

    # 4. Audit & Publish 2.3
    print(f"Running audit script for gen 2.3...", flush=True)
    run_cmd(["rm", "-rf", "open_spiel/python/games/backgammon/audit_temp/"])
    run_cmd(["python3", "open_spiel/python/games/backgammon/gnubg_auditor/generational_auditor.py", "2.3"])

    # Archive Gen 2.2 Collection Data
    print("Archiving Gen 2.2 collection games...", flush=True)
    archive_cmd_2 = [
        "python3", "-c",
        "from google.cloud import storage; "
        "client = storage.Client(); "
        "bucket = client.bucket('expert-eyes-training-742'); "
        "blobs = bucket.list_blobs(prefix='logs/game_gen2_mass_collection_target'); "
        "[(bucket.rename_blob(b, b.name.replace('logs/', 'archive/gen_2.2/', 1))) for b in blobs if not b.name.endswith('/')]"
    ]
    run_cmd(archive_cmd_2)

    print("Phase 2 Pipeline completed successfully.", flush=True)

if __name__ == "__main__":
    print("Starting Phase 2 Automated Loop...", flush=True)
    execute_phase2()
