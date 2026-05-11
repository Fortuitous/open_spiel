#!/usr/bin/env python3
"""
Master Orchestrator for Expert Eyes Backgammon Training Loop.

Runs locally on the Ubuntu workstation. Submits Vertex AI jobs via gcloud CLI,
polls for completion, runs gnubg audits locally, and updates Google Sheets.

MODEL-FIRST NAMING CONVENTION:
  - Collection N   = 1000 games played USING Model N weights.
  - Training N     = Training ON Collection N data to PRODUCE Model N+0.1.
  - Audit N+0.1    = 20 diagnostic games played with new Model N+0.1 weights.

CYCLE (starting from Model N, already audited):
  1. Collection N         (Vertex AI, 8 workers)
  2. Training N           (Vertex AI, 1 worker → produces Model N+0.1)
  3. Archive Collection N (gsutil mv to archive/)
  4. Audit N+0.1          (Vertex AI, 2 jobs: greedy + MCTS)
  5. Archive audit transcripts
  6. GNUbg analysis       (local)
  7. Google Sheets update
  8. Save state as N+0.1, loop to step 1

Usage:
    python3 master_orchestrator.py                    # Resume from GCS state
    python3 master_orchestrator.py --start-gen 2.2    # Force start from gen 2.2
    python3 master_orchestrator.py --dry-run           # Print without executing

Graceful shutdown:
    touch STOP              # Script finishes current cycle then exits
    Ctrl+C                  # Same: finishes current cycle then exits
    Ctrl+C twice            # Immediate abort
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.parse
from datetime import datetime

# ──────────────────────────── Configuration ────────────────────────────

PROJECT_ID = "expert-eyes-training-742"
REGION = "us-central1"
BUCKET = "expert-eyes-training-742"
IMAGE_URI = ("us-central1-docker.pkg.dev/expert-eyes-training-742/"
             "expert-eyes-repo/trainer:v21")
SHEET_ID = "1vmbqRunR5UI-ZUnEulx5-apuCdL0MY4av7miGVpH5aI"
SHEETS_SA_KEY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".sheets-sa-key.json")
WANDB_KEY = ("wandb_v1_7MK68vKCpzhagZo1rCYFcShbqqV_ZHpAfIussAQx3Oliu"
             "eyInrBWlxQgy5hmoRrKm2h5z8X1wLfwI")

# Dynamic generation-based prefix for collection data
def get_collection_prefix(gen):
    return f"gen{gen_str(gen)}_mass_collection"

# Legacy/Base prefix if needed
COLLECTION_PREFIX = "gen2_mass_collection_target"
AUDITOR_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "open_spiel/python/games/backgammon/gnubg_auditor/generational_auditor.py")

# Polling intervals (seconds)
POLL_COLLECT  = 600   # 10 min
POLL_TRAIN    = 300   #  5 min
POLL_DIAG     = 180   #  3 min

DRY_RUN = False
STOP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STOP")
SHUTDOWN_REQUESTED = False

def _handle_sigint(signum, frame):
    global SHUTDOWN_REQUESTED
    if SHUTDOWN_REQUESTED:
        log("Second interrupt received. Aborting immediately.")
        sys.exit(1)
    SHUTDOWN_REQUESTED = True
    log("")
    log("╔══════════════════════════════════════════════════════════╗")
    log("║  SHUTDOWN REQUESTED — will exit after current cycle.    ║")
    log("║  Press Ctrl+C again to abort immediately.               ║")
    log("╚══════════════════════════════════════════════════════════╝")

signal.signal(signal.SIGINT, _handle_sigint)

def should_stop():
    """Check if a graceful stop has been requested."""
    global SHUTDOWN_REQUESTED
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
        SHUTDOWN_REQUESTED = True
        log("STOP file detected. Will exit after current cycle.")
    return SHUTDOWN_REQUESTED

# ──────────────────────────── Helpers ──────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run(cmd):
    """Run a shell command, return (stdout, stderr, returncode)."""
    log(f"  CMD: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode

def gen_str(gen):
    """Format a generation float as a clean string: 2.0, 2.1, etc."""
    return f"{gen:.1f}" if gen == int(gen) or (gen * 10) % 1 == 0 else f"{gen}"

# ──────────────────────────── GCS State ────────────────────────────────

def load_gen_state():
    stdout, _, rc = run(["gsutil", "cat", f"gs://{BUCKET}/current_gen.json"])
    if rc == 0:
        try:
            return float(json.loads(stdout).get("current_gen", 2.0))
        except (json.JSONDecodeError, ValueError):
            pass
    log("WARNING: Could not read current_gen.json, defaulting to 2.0")
    return 2.0

def save_gen_state(gen):
    data = json.dumps({"current_gen": gen, "updated": datetime.now().isoformat()})
    if DRY_RUN:
        log(f"  [DRY-RUN] Would save gen state → {gen_str(gen)}")
        return
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(data)
        tmp = f.name
    run(["gsutil", "cp", tmp, f"gs://{BUCKET}/current_gen.json"])
    os.remove(tmp)
    log(f"Saved gen state → {gen_str(gen)}")

# ──────────────────────────── YAML Generation ─────────────────────────

def _wandb_env():
    return (f'      env:\n'
            f'        - name: WANDB_RUN_GROUP\n'
            f'          value: "gen2-pipeline"\n'
            f'        - name: WANDB_API_KEY\n'
            f'          value: "{WANDB_KEY}"')

def yaml_collection(gen):
    prefix = get_collection_prefix(gen)
    worker = (
        f'  - machineSpec:\n'
        f'      machineType: n1-highmem-4\n'
        f'      acceleratorType: NVIDIA_TESLA_T4\n'
        f'      acceleratorCount: 1\n'
        f'    replicaCount: {{replicas}}\n'
        f'    containerSpec:\n'
        f'      imageUri: {IMAGE_URI}\n'
        f'      args:\n'
        f'        - "--mode"\n'
        f'        - "play"\n'
        f'        - "--num_games"\n'
        f'        - "125"\n'
        f'        - "--global_target"\n'
        f'        - "1000"\n'
        f'        - "--job_id"\n'
        f'        - "{prefix}"\n'
        f'        - "--bucket_name"\n'
        f'        - "{BUCKET}"\n'
        f'        - "--wandb"\n'
        f'{_wandb_env()}')
    w0 = worker.format(replicas=1)
    w1 = worker.format(replicas=7)
    return (f'workerPoolSpecs:\n{w0}\n{w1}\n'
            f'scheduling:\n  timeout: 43200s\n'
            f'  restartJobOnWorkerRestart: true\n  strategy: SPOT\n')

def yaml_train(gen):
    return (
        f'workerPoolSpecs:\n'
        f'  - machineSpec:\n'
        f'      machineType: n1-standard-4\n'
        f'      acceleratorType: NVIDIA_TESLA_T4\n'
        f'      acceleratorCount: 1\n'
        f'    replicaCount: 1\n'
        f'    containerSpec:\n'
        f'      imageUri: {IMAGE_URI}\n'
        f'      args:\n'
        f'        - "--mode"\n'
        f'        - "train"\n'
        f'        - "--lr"\n'
        f'        - "0.001"\n'
        f'        - "--batch_size"\n'
        f'        - "256"\n'
        f'        - "--steps"\n'
        f'        - "5000"\n'
        f'        - "--bucket_name"\n'
        f'        - "{BUCKET}"\n'
        f'        - "--log_prefix"\n'
        f'        - "logs/game_{get_collection_prefix(gen)}"\n'
        f'        - "--wandb"\n'
        f'{_wandb_env()}\n'
        f'scheduling:\n  timeout: 14400s\n'
        f'  restartJobOnWorkerRestart: true\n  strategy: SPOT\n')

def yaml_diag(gen, sims, job_id):
    return (
        f'workerPoolSpecs:\n'
        f'  - machineSpec:\n'
        f'      machineType: n1-highmem-8\n'
        f'      acceleratorType: NVIDIA_TESLA_T4\n'
        f'      acceleratorCount: 1\n'
        f'    replicaCount: 1\n'
        f'    containerSpec:\n'
        f'      imageUri: {IMAGE_URI}\n'
        f'      args:\n'
        f'        - "--mode=play"\n'
        f'        - "--num_games=10"\n'
        f'        - "--sims={sims}"\n'
        f'        - "--bucket_name={BUCKET}"\n'
        f'        - "--job_id={job_id}"\n')

# ──────────────────────────── Job Management ──────────────────────────

def write_temp_yaml(content):
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=".") as f:
        f.write(content)
        return f.name

def submit_job(display_name, yaml_content):
    """Submit a Vertex AI custom job. Returns the job resource name."""
    if DRY_RUN:
        log(f"  [DRY-RUN] Would submit: {display_name}")
        return "dry-run-job"

    cfg = write_temp_yaml(yaml_content)
    try:
        stdout, stderr, rc = run([
            "gcloud", "ai", "custom-jobs", "create",
            f"--project={PROJECT_ID}", f"--region={REGION}",
            f"--display-name={display_name}", f"--config={cfg}"])
    finally:
        os.remove(cfg)

    combined = stdout + "\n" + stderr
    match = re.search(
        r"projects/\d+/locations/[^/]+/customJobs/\d+", combined)
    if match:
        job_name = match.group(0)
        log(f"  Submitted: {job_name}")
        return job_name

    log(f"  FAILED to submit {display_name}:\n{combined}")
    return None

def wait_for_job(job_name, poll_interval):
    """Poll a Vertex AI job until it finishes. Returns True on success."""
    if DRY_RUN:
        log("  [DRY-RUN] Would poll job until completion.")
        return True

    while True:
        stdout, stderr, rc = run([
            "gcloud", "ai", "custom-jobs", "describe", job_name,
            "--format=json"])
        if rc != 0:
            log(f"  Error polling: {stderr}")
            time.sleep(poll_interval)
            continue
        try:
            state = json.loads(stdout).get("state", "UNKNOWN")
        except json.JSONDecodeError:
            log(f"  Bad JSON from describe: {stdout[:200]}")
            time.sleep(poll_interval)
            continue

        log(f"  Job state: {state}")
        if state == "JOB_STATE_SUCCEEDED":
            return True
        if state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                      "JOB_STATE_EXPIRED"):
            log(f"  Job FAILED with state: {state}")
            return False
        time.sleep(poll_interval)

# ──────────────────────────── Archive ─────────────────────────────────

def archive_collection_data(gen):
    """Move mass-collection .pt_data and _xg.txt from logs/ to archive/."""
    gs = f"gs://{BUCKET}"
    dest = f"{gs}/archive/gen_{gen_str(gen)}/"
    log(f"Archiving collection data → {dest}")
    if DRY_RUN:
        return
    # Move both .pt_data and _xg.txt files
    prefix = get_collection_prefix(gen)
    run(["gsutil", "-m", "mv",
         f"{gs}/logs/game_{prefix}_*", dest])

def archive_audit_transcripts(audit_gen):
    """Copy diagnostic _xg.txt files to audits/gen_X_transcripts/."""
    gs = f"gs://{BUCKET}"
    g = gen_str(audit_gen)
    dest = f"{gs}/audits/gen_{g}_transcripts/"
    log(f"Archiving audit transcripts → {dest}")
    if DRY_RUN:
        return
    for prefix in [f"game_diag_gen{g}_no_mcts_", f"game_diag_gen{g}_mcts_"]:
        run(["gsutil", "-m", "cp",
             f"{gs}/logs/{prefix}*_xg.txt", dest])

# ──────────────────────────── Local Audit Copy ─────────────────────────

WINDOWS_AUDIT_DIR = "/mnt/c/Users/jerem/Documents/ExpertEyes"

def download_audits_locally(audit_gen):
    """Download audit transcripts from GCS to the Windows filesystem."""
    g = gen_str(audit_gen)
    log(f"Downloading Audit {g} transcripts to Windows...")
    if DRY_RUN:
        log("  [DRY-RUN] Would download to Windows filesystem")
        return

    for label, prefix in [("Greedy", f"game_diag_gen{g}_no_mcts_"),
                          ("MCTS",   f"game_diag_gen{g}_mcts_")]:
        dest = os.path.join(WINDOWS_AUDIT_DIR, f"Audit_{g}", label)
        os.makedirs(dest, exist_ok=True)
        run(["gsutil", "-m", "cp",
             f"gs://{BUCKET}/logs/{prefix}*_xg.txt", dest])

    log(f"  Saved to {WINDOWS_AUDIT_DIR}/Audit_{g}/")

# ──────────────────────────── Audit (local) ───────────────────────────

def run_gnubg_audit(audit_gen):
    """Run generational_auditor.py locally."""
    g = gen_str(audit_gen)
    log(f"Running GNUbg audit for gen {g}...")
    if DRY_RUN:
        log("  [DRY-RUN] Would run generational_auditor.py")
        return True
    rc = subprocess.run(
        ["python3", AUDITOR_SCRIPT, g],
        cwd=os.path.dirname(os.path.abspath(__file__))).returncode
    if rc != 0:
        log(f"  Auditor exited with code {rc}")
        return False
    return True

# ──────────────────────────── Google Sheets ───────────────────────────

def get_access_token():
    """Get an OAuth2 token from the service account key file."""
    try:
        with open(SHEETS_SA_KEY) as f:
            sa = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        log(f"  Could not read SA key: {e}")
        return None

    # Build JWT
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "RS256", "typ": "JWT"}).encode()).rstrip(b"=")
    now = int(time.time())
    payload = base64.urlsafe_b64encode(json.dumps({
        "iss": sa["client_email"],
        "scope": ("https://www.googleapis.com/auth/spreadsheets "
                 "https://www.googleapis.com/auth/drive.file"),
        "aud": "https://oauth2.googleapis.com/token",
        "iat": now, "exp": now + 3600
    }).encode()).rstrip(b"=")

    # Sign with RSA-SHA256 using the service account private key
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        key = serialization.load_pem_private_key(
            sa["private_key"].encode(), password=None)
        signature = key.sign(
            header + b"." + payload,
            padding.PKCS1v15(), hashes.SHA256())
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=")
    except ImportError:
        # Fallback: use openssl CLI
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pem",
                                         delete=False) as kf:
            kf.write(sa["private_key"])
            key_path = kf.name
        try:
            proc = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", key_path],
                input=header + b"." + payload,
                capture_output=True)
            sig_b64 = base64.urlsafe_b64encode(proc.stdout).rstrip(b"=")
        finally:
            os.remove(key_path)

    jwt_token = header + b"." + payload + b"." + sig_b64

    # Exchange JWT for access token
    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=urllib.parse.urlencode({
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt_token.decode()
        }).encode())
    try:
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read())["access_token"]
    except Exception as e:
        log(f"  Token exchange failed: {e}")
        return None

def _sheets_request(headers, endpoint, body):
    url = f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}:{endpoint}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        urllib.request.urlopen(req)
    except Exception:
        pass  # Tab may already exist

def _sheets_append(headers, sheet_name, rows):
    if not rows:
        return
    encoded = urllib.parse.quote(sheet_name)
    url = (f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}"
           f"/values/{encoded}:append?valueInputOption=RAW")
    req = urllib.request.Request(
        url, data=json.dumps({"values": rows}).encode(),
        headers=headers, method="POST")
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        log(f"  Sheet append error: {e}")

def sheet_log(message, level="INFO"):
    """Append a timestamped log row to the 'Logs' tab in Google Sheet."""
    if DRY_RUN:
        return
    token = get_access_token()
    if not token:
        return
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}
    # Ensure Logs tab exists
    _sheets_request(headers, "batchUpdate", {
        "requests": [{"addSheet": {"properties": {"title": "Logs"}}}]})
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _sheets_append(headers, "Logs", [[ts, level, message]])

def update_google_sheet(audit_gen):
    """Read audit results from GCS and append to Google Sheet."""
    g = gen_str(audit_gen)
    log(f"Updating Google Sheet for gen {g}...")
    if DRY_RUN:
        log("  [DRY-RUN] Would update sheet.")
        return

    token = get_access_token()
    if not token:
        log("  WARNING: Could not get access token. Skipping sheet update.")
        return

    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}

    # Download audit run file from GCS
    stdout, _, rc = run(["gsutil", "cat",
                         f"gs://{BUCKET}/summaries/audit_run_{g}.txt"])
    if rc != 0:
        log(f"  Could not download audit_run_{g}.txt")
        return

    # Parse rows into a table
    rows = []
    for line in stdout.strip().split("\n"):
        if line.strip():
            rows.append([cell.strip() for cell in line.split("/")])

    # Create a tab for this generation
    tab_name = f"{g} Audit"
    _sheets_request(headers, "batchUpdate", {
        "requests": [{"addSheet": {"properties": {"title": tab_name}}}]})

    # Append data
    _sheets_append(headers, tab_name, rows)

    # Also update Master tab with the summary line from master_audit_summary
    stdout2, _, rc2 = run(["gsutil", "cat",
                           f"gs://{BUCKET}/summaries/master_audit_summary.txt"])
    if rc2 == 0 and stdout2.strip():
        # Get only the last line (the one just appended for this gen).
        # Format: "2.3 / Greedy / ... / 41         2.3 / MCTS / ... / 42"
        # Greedy and MCTS are on the same line separated by whitespace.
        # Split into two separate rows for the sheet.
        last_line = stdout2.strip().split("\n")[-1]
        # Split on multiple spaces/tabs to separate Greedy and MCTS halves
        import re as _re
        halves = _re.split(r"\s{2,}", last_line.strip())
        master_rows = []
        for half in halves:
            if half.strip():
                master_rows.append([cell.strip() for cell in half.split("/")])
        _sheets_append(headers, "Master", master_rows)

    log(f"  Sheet updated for {g}.")

# ──────────────────────────── Main Cycle ──────────────────────────────

def run_cycle(gen):
    """
    Run one full cycle starting from Model `gen` (already audited).

    1. Collection gen       →  1000 games with Model gen
    2. Training gen         →  produces Model gen+0.1
    3. Archive collection data
    4. Audit gen+0.1        →  20 diagnostic games with new model
    5. Archive audit transcripts
    6. GNUbg + Sheets
    7. Return gen+0.1
    """
    g = gen_str(gen)
    next_gen = round(gen + 0.1, 1)
    ng = gen_str(next_gen)

    # ── Step 1: Collection ──
    log(f"═══ STEP 1: Collection {g} (1000 games with Model {g}) ═══")
    sheet_log(f"Starting Collection {g} (1000 games with Model {g})")
    
    # Pre-collection cleanup to ensure no old files confuse the count
    prefix = get_collection_prefix(gen)
    log(f"  Cleaning up any existing logs with prefix {prefix}...")
    run(["gsutil", "-m", "rm", "-f", f"gs://{BUCKET}/logs/game_{prefix}_*"])
    
    job = submit_job(f"collection-{g}", yaml_collection(gen))
    if not job or not wait_for_job(job, POLL_COLLECT):
        sheet_log(f"FAILED: Collection {g} job failed", "ERROR")
        log("FATAL: Collection failed."); sys.exit(1)
    sheet_log(f"Completed Collection {g}")

    # ── Step 2: Training ──
    log(f"═══ STEP 2: Training {g} (→ Model {ng}) ═══")
    sheet_log(f"Starting Training {g} (→ Model {ng})")
    job = submit_job(f"training-{g}", yaml_train(gen))
    if not job or not wait_for_job(job, POLL_TRAIN):
        sheet_log(f"FAILED: Training {g} job failed", "ERROR")
        log("FATAL: Training failed."); sys.exit(1)
    sheet_log(f"Completed Training {g} → Model {ng} created")

    # ── Step 3: Archive collection data ──
    log(f"═══ STEP 3: Archive Collection {g} data ═══")
    sheet_log(f"Archiving Collection {g} data")
    archive_collection_data(gen)

    # ── Step 4: Audit with new model ──
    log(f"═══ STEP 4: Audit {ng} (20 games with Model {ng}) ═══")
    sheet_log(f"Starting Audit {ng} (20 games with Model {ng})")
    greedy_id = f"diag_gen{ng}_no_mcts"
    mcts_id = f"diag_gen{ng}_mcts_400"

    j1 = submit_job(f"audit-{ng}-greedy", yaml_diag(ng, 1, greedy_id))
    j2 = submit_job(f"audit-{ng}-mcts",   yaml_diag(ng, 400, mcts_id))

    if j1 and not wait_for_job(j1, POLL_DIAG):
        sheet_log(f"FAILED: Audit {ng} greedy job failed", "ERROR")
        log("FATAL: Greedy audit job failed."); sys.exit(1)
    if j2 and not wait_for_job(j2, POLL_DIAG):
        sheet_log(f"FAILED: Audit {ng} MCTS job failed", "ERROR")
        log("FATAL: MCTS audit job failed."); sys.exit(1)
    sheet_log(f"Completed Audit {ng} eval games")

    # ── Step 5: Archive audit transcripts ──
    log(f"═══ STEP 5: Archive Audit {ng} transcripts ═══")
    archive_audit_transcripts(next_gen)

    # ── Step 5b: Download audit games to Windows ──
    log(f"═══ STEP 5b: Download Audit {ng} to Windows ═══")
    sheet_log(f"Downloading Audit {ng} transcripts to Windows")
    download_audits_locally(next_gen)

    # ── Step 6: GNUbg analysis (local) ──
    log(f"═══ STEP 6: GNUbg Analysis for {ng} ═══")
    sheet_log(f"Running GNUbg analysis for Audit {ng}")
    run_gnubg_audit(next_gen)

    # ── Step 7: Google Sheets ──
    log(f"═══ STEP 7: Google Sheets update for {ng} ═══")
    update_google_sheet(next_gen)

    # ── Step 8: Save state ──
    save_gen_state(next_gen)
    sheet_log(f"✓ Cycle complete: Model {g} → Model {ng}")
    log(f"═══ Cycle complete: Model {g} → Model {ng} ═══\n")

    return next_gen

# ──────────────────────────── Entry Point ─────────────────────────────

def main():
    global DRY_RUN

    parser = argparse.ArgumentParser(
        description="Master Orchestrator for Expert Eyes Training Loop")
    parser.add_argument("--start-gen", type=float, default=None,
                        help="Force start from this generation (e.g. 2.2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without executing")
    parser.add_argument("--single-cycle", action="store_true",
                        help="Run one cycle then exit")
    args = parser.parse_args()

    DRY_RUN = args.dry_run

    if args.start_gen is not None:
        gen = args.start_gen
        log(f"Starting from forced gen: {gen_str(gen)}")
    else:
        gen = load_gen_state()
        log(f"Resuming from GCS state: {gen_str(gen)}")

    log(f"{'='*60}")
    log(f"  Expert Eyes Master Orchestrator")
    log(f"  Starting gen:  {gen_str(gen)}")
    log(f"  Image:         {IMAGE_URI}")
    log(f"  Dry run:       {DRY_RUN}")
    log(f"{'='*60}\n")

    while True:
        gen = run_cycle(gen)
        if args.single_cycle:
            log("Single-cycle mode. Exiting.")
            break
        if should_stop():
            log("Graceful shutdown. Exiting after completed cycle.")
            break
        log("Sleeping 60s before next cycle...\n")
        time.sleep(60)
        if should_stop():
            log("Graceful shutdown. Exiting.")
            break

if __name__ == "__main__":
    main()
