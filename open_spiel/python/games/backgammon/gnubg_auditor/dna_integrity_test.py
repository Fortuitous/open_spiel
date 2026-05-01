import subprocess
import sys
import os
from google.cloud import storage

def dna_integrity_test(bucket_name, log_prefix):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # List logs
    blobs = list(client.list_blobs(bucket, prefix=log_prefix))
    if not blobs:
        print(f"No logs found in gs://{bucket_name}/{log_prefix}")
        return False
        
    # Pick the most recent XG log
    xg_blobs = [b for b in blobs if b.name.endswith("_xg.txt")]
    if not xg_blobs:
        print("No _xg.txt logs found.")
        return False
        
    xg_blobs.sort(key=lambda x: x.updated, reverse=True)
    target_blob = xg_blobs[0]
    local_file = "test_dna.txt"
    target_blob.download_to_filename(local_file)
    print(f"Downloaded {target_blob.name} for testing.")

    # Import into GNUbg
    cmd_file = "dna_test.p"
    with open(cmd_file, "w") as f:
        f.write(f"import auto {os.path.abspath(local_file)}\n")
        f.write("analyze match\n")
        f.write("show statistics\n")
        f.write("quit\n")

    cmd = f"/usr/games/gnubg -t -q < {cmd_file}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if "Overall error rate" in res.stdout or "Checkerplay error rate" in res.stdout:
        print("✅ DNA INTEGRITY VERIFIED: GNUbg successfully analyzed the log.")
        # Print a snippet of the stats
        for line in res.stdout.split("\n"):
            if "error rate" in line.lower():
                print(f"  {line.strip()}")
        return True
    else:
        print("❌ DNA INTEGRITY FAILED: GNUbg could not analyze the log.")
        print("STDOUT:", res.stdout[:500])
        return False

if __name__ == "__main__":
    dna_integrity_test("expert-eyes-training-742", "logs/")
