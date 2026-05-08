import os
import json
import subprocess

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "expert-eyes-training-742")
SHEET_ID = os.environ.get("SHEET_ID")

def submit_vertex_job():
    print("Submitting Phase 1 Vertex AI CustomJob for Gen 2.0 Baseline Initialization...")

    script_content = f"""
import subprocess
import urllib.request
import json
import os
import sys

print("--- 1. Initializing Weights ---")
try:
    subprocess.run(["python3", "open_spiel/python/games/backgammon/trainer_v1.py", "--mode", "init_scratch", "--bucket_name", "expert-eyes-training-742"], check=True)
except Exception as e:
    print("Error in init_scratch:", e)

print("--- 2. Eval Collect ---")
try:
    subprocess.run(["python3", "open_spiel/python/games/backgammon/trainer_v1.py", "--mode", "play", "--num_games", "10", "--sims", "1", "--bucket_name", "expert-eyes-training-742", "--job_id", "diag_gen2.0_no_mcts"], check=True)
except Exception as e:
    print("Error in eval 1:", e)
try:
    subprocess.run(["python3", "open_spiel/python/games/backgammon/trainer_v1.py", "--mode", "play", "--num_games", "10", "--sims", "400", "--bucket_name", "expert-eyes-training-742", "--job_id", "diag_gen2.0_mcts"], check=True)
except Exception as e:
    print("Error in eval 2:", e)

print("--- 3. Audit ---")
try:
    subprocess.run(["python3", "open_spiel/python/games/backgammon/gnubg_auditor/generational_auditor.py", "2.0"], check=True)
except Exception as e:
    print("Error in audit:", e)

print("--- 4. Publish ---")
from google.cloud import storage
import google.auth
from google.auth.transport.requests import Request
client = storage.Client()
bucket = client.bucket("expert-eyes-training-742")

try:
    audit_run_text = bucket.blob("summaries/audit_run_2.0.txt").download_as_text()
    master_audit_text = bucket.blob("summaries/master_audit_summary.txt").download_as_text()
except Exception as e:
    print(f"Failed to download summaries: {{e}}")
    audit_run_text = ""
    master_audit_text = ""

def parse_text(text):
    rows = []
    for line in text.strip().split('\\n'):
        if line.strip():
            rows.append([cell.strip() for cell in line.split('/')])
    return rows

audit_run_data = parse_text(audit_run_text)
master_audit_data = parse_text(master_audit_text)

SHEET_ID = "{SHEET_ID}"
if not SHEET_ID:
    print("SHEET_ID is missing")
else:
    try:
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/spreadsheets"])
        req = Request()
        creds.refresh(req)
        access_token = creds.token
    except Exception as e:
        print("Failed to get credentials:", e)
        sys.exit(0) # Do not crash the job, we already completed the run.

    headers = {{"Authorization": f"Bearer {{access_token}}", "Content-Type": "application/json"}}

    batch_update_url = f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}:batchUpdate"
    add_sheets_payload = {{"requests": [{{"addSheet": {{"properties": {{"title": "2.0 Audit"}}}}}}, {{"addSheet": {{"properties": {{"title": "Master"}}}}}}]}}

    req_create = urllib.request.Request(batch_update_url, data=json.dumps(add_sheets_payload).encode("utf-8"), headers=headers, method="POST")
    try:
        urllib.request.urlopen(req_create)
    except Exception as e:
        print(f"Sheets might already exist: {{e}}")

    def append_data(sheet_name, data):
        if not data: return
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/'{{sheet_name}}':append?valueInputOption=RAW"
        req_append = urllib.request.Request(url, data=json.dumps({{"values": data}}).encode("utf-8"), headers=headers, method="POST")
        try:
            urllib.request.urlopen(req_append)
            print(f"Appended to {{sheet_name}}.")
        except Exception as e:
            print(f"Error appending: {{e}}")

    append_data("2.0 Audit", audit_run_data)
    append_data("Master", master_audit_data)

print("=== PHASE 1 COMPLETE ===")
"""
    config = {
        "workerPoolSpecs": [
            {
                "machineSpec": {
                    "machineType": "n1-highmem-8",
                    "acceleratorType": "NVIDIA_TESLA_T4",
                    "acceleratorCount": 1
                },
                "replicaCount": 1,
                "containerSpec": {
                    "imageUri": "us-central1-docker.pkg.dev/expert-eyes-training-742/expert-eyes-repo/trainer:v19",
                    "command": ["python3", "-c", script_content],
                    "env": [
                        {"name": "SHEET_ID", "value": SHEET_ID}
                    ]
                }
            }
        ]
    }

    with open("job_spec.json", "w") as f:
        json.dump(config, f)

    cmd = [
        "gcloud", "ai", "custom-jobs", "create",
        f"--project={PROJECT_ID}",
        "--region=us-central1",
        "--display-name=phase-1-gen2-baseline",
        "--config=job_spec.json"
    ]

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", proc.stdout)
    if proc.stderr:
        print("STDERR:", proc.stderr)

    if proc.returncode != 0:
        raise Exception("Failed to submit job")

    # Get the job ID from stdout
    import re
    match = re.search(r'projects/\d+/locations/us-central1/customJobs/\d+', proc.stdout)
    if not match:
        match = re.search(r'projects/\d+/locations/us-central1/customJobs/\d+', proc.stderr)

    if match:
        job_name = match.group(0)
        print(f"Job submitted successfully! We will monitor it. Job Name: {job_name}")

if __name__ == "__main__":
    submit_vertex_job()
