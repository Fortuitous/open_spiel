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

    # We are resuming from Step 3: Eval Collect for Gen 2.1 (which outputs Model 2.2 logs)
    greedy_job = "projects/947210424180/locations/us-central1/customJobs/332487076379361280"
    mcts_job = "projects/947210424180/locations/us-central1/customJobs/862785932502237184"

    # Ensure they are done
    wait_for_job(greedy_job, poll_interval=600)
    wait_for_job(mcts_job, poll_interval=600)

    # 4. Audit & Publish (This evaluates Gen 2.2 model produced from Gen 2.1 training)
    print(f"Running audit script for gen 2.2...", flush=True)
    run_cmd(["python3", "open_spiel/python/games/backgammon/gnubg_auditor/generational_auditor.py", "2.2"])

    # 5. Archive Collection Data
    print("Archiving Gen 2.1 collection games...")
    archive_cmd = [
        "python3", "-c",
        "from google.cloud import storage; "
        "client = storage.Client(); "
        "bucket = client.bucket('expert-eyes-training-742'); "
        "blobs = bucket.list_blobs(prefix='logs/game_gen2_mass_collection_target'); "
        "[(bucket.rename_blob(b, b.name.replace('logs/', 'archive/gen_2.1/', 1))) for b in blobs if not b.name.endswith('/')]"
    ]
    run_cmd(archive_cmd)

    print(f"Phase 1 Pipeline completed successfully. To enter Phase 2 loop, execute with gen_name=2.2", flush=True)
    return True

if __name__ == "__main__":
    print("Resuming automated loop from Audit 2.2 phase...", flush=True)
    execute_pipeline("2.1")
