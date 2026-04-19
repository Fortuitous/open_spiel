import subprocess
import re
import os
import sys
from google.cloud import storage

def clean_legacy_log(content):
    """
    Fixes the (N) notation bug for non-double rolls in v17 logs.
    Also ensures spacing is more robust for GNUbg CLI import.
    """
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        # Improved regex to handle Bar, Off, and Hits (*)
        # Matches: "  19) 65: Bar/20* Bar/19              43: Bar/22"
        match = re.search(r"^\s*(\d+)\)\s+(\d)(\d): (.*?)\s{2,}(\d)(\d): (.*)$", line)
        if match:
            idx, d1, d2, m1, d3, d4, m2 = match.groups()
            
            # Helper to expand move(N) notation
            def expand(m):
                return re.sub(r"(\S+)\((\d)\)", lambda sub: (sub.group(1) + " ") * int(sub.group(2)), m).strip()

            if d1 != d2: m1 = expand(m1)
            if d3 != d4: m2 = expand(m2)
            
            # Snowie format is picky about columns. 
            # We'll use a safer fixed-width reconstruction.
            new_line = f"  {idx:>2}) {d1}{d2}: {m1:<28} {d3}{d4}: {m2}"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

def parse_gnubg_stats(stdout):
    """Parses gnubg output by specifically looking at the Overall Statistics section."""
    luck_section = re.search(r"Luck statistics.*?(?=Cube statistics|$)", stdout, re.DOTALL | re.IGNORECASE)
    overall_section = re.search(r"Overall statistics.*?(?=Final score|$)", stdout, re.DOTALL | re.IGNORECASE)
    
    combined_text = (luck_section.group(0) if luck_section else "") + (overall_section.group(0) if overall_section else "")
    if not combined_text:
        combined_text = stdout

    patterns = {
        'checker_error': r"Error rate mEMG \(MWC\)\s+(-?\d+\.\d+)\s+\(.*?%\)\s+(-?\d+\.\d+)",
        'snowie_error': r"Snowie error rate\s+(-?\d+\.\d+)\s+\(.*?%\)\s+(-?\d+\.\d+)",
        'luck_total': r"Luck total EMG \(MWC\)\s+([-+]\d+\.\d+)\s+\(.*?%\)\s+([-+]\d+\.\d+)",
        'move_count': r"Total moves\s+(\d+)\s+(\d+)"
    }
    
    parsed = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, combined_text, re.DOTALL)
        if not match:
            match = re.search(pattern, stdout, re.DOTALL)
            
        if match:
            parsed[key] = match.groups()
        else:
            parsed[key] = [0.0, 0.0]

    results = []
    for i in range(2):
        results.append({
            'checker_error': float(parsed['checker_error'][i]),
            'snowie_error': float(parsed['snowie_error'][i]),
            'luck': float(parsed['luck_total'][i]),
            'moves': int(parsed['move_count'][i])
        })
    return results

def analyze_game(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    cleaned_content = clean_legacy_log(content)
    with open(file_path, 'w') as f:
        f.write(cleaned_content)

    cmd = ["/usr/games/gnubg", "-t", "-q"]
    # Explicitly set 2-ply analysis to match user GUI settings
    stdin_content = (
        f"import auto {file_path}\n"
        "set analysis evaluation plies 2\n"
        "analyze match\n"
        "show statistics match\n"
        "quit\n"
    )
    
    try:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(input=stdin_content, timeout=120) # Increased timeout for 2-ply
    except subprocess.TimeoutExpired:
        return None

    return parse_gnubg_stats(stdout)

def run_sampled_audit(prefix="gen1.0", num_batches=10, games_per_batch=10):
    client = storage.Client()
    bucket = client.bucket("expert-eyes-training-742")
    
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
        
    blobs = list(client.list_blobs(bucket, prefix=f"logs/game_{prefix}_"))
    xg_blobs = [b for b in blobs if b.name.endswith("_xg.txt")]
    xg_blobs.sort(key=lambda b: natural_sort_key(b.name))
    
    if not xg_blobs:
        print("No logs found.")
        return

    print(f"Found {len(xg_blobs)} games. Starting Cleaned 2-Ply Sampled Audit...")
    
    total_games = len(xg_blobs)
    if num_batches > 1:
        batch_indices = [int(i * (total_games - games_per_batch) / (num_batches - 1)) for i in range(num_batches)]
    else:
        batch_indices = [0]
    
    all_observations = []
    
    print("| Batch | Game | Player | Checker Error | Snowie Error | Luck | Moves |")
    print("|-------|------|--------|---------------|--------------|------|-------|")
    
    for b_idx, start_idx in enumerate(batch_indices):
        batch = xg_blobs[start_idx : start_idx + games_per_batch]
        for g_idx, blob in enumerate(batch):
            local_path = f"temp_audit_{b_idx}_{g_idx}.txt"
            blob.download_to_filename(local_path)
            
            res = analyze_game(os.path.abspath(local_path))
            os.remove(local_path)
            
            if res:
                for p_idx, obs in enumerate(res):
                    print(f"| {b_idx+1} | {start_idx+g_idx} | P{p_idx+1} | {obs['checker_error']:.1f} | {obs['snowie_error']:.1f} | {obs['luck']:.3f} | {obs['moves']} |")
                    all_observations.append(obs)

    if all_observations:
        checker_errors = [o['checker_error'] for o in all_observations]
        snowie_errors = [o['snowie_error'] for o in all_observations]
        
        avg_checker = sum(checker_errors) / len(checker_errors)
        avg_snowie = sum(snowie_errors) / len(snowie_errors)
        
        sorted_checker = sorted(checker_errors)
        median_checker = sorted_checker[len(sorted_checker)//2]
        
        sorted_snowie = sorted(snowie_errors)
        median_snowie = sorted_snowie[len(sorted_snowie)//2]
        
        print("\n### Cleaned 2-Ply Audit Summary")
        print(f"* **Average Checker Error**: {avg_checker:.2f}")
        print(f"* **Median Checker Error**: {median_checker:.2f}")
        print(f"* **Average Snowie Error**: {avg_snowie:.2f}")
        print(f"* **Median Snowie Error**: {median_snowie:.2f}")

if __name__ == "__main__":
    import sys
    prefix = sys.argv[1] if len(sys.argv) > 1 else "gen1.0"
    num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    games_per_batch = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    run_sampled_audit(prefix=prefix, num_batches=num_batches, games_per_batch=games_per_batch)
