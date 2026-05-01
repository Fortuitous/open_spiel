import subprocess
import sys
import os
import json
from google.cloud import storage

def batch_audit(bucket_name, prefix, limit=100):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    xg_blobs = [b for b in blobs if b.name.endswith("_xg.txt")]
    xg_blobs.sort(key=lambda x: x.updated, reverse=True)
    xg_blobs = xg_blobs[:limit]
    
    print(f"Found {len(xg_blobs)} logs to audit.")
    
    results = []
    os.makedirs("audit_temp", exist_ok=True)
    
    for b in xg_blobs:
        local_file = os.path.join("audit_temp", os.path.basename(b.name))
        b.download_to_filename(local_file)
        
        cmd_file = "audit_batch.p"
        with open(cmd_file, "w") as f:
            f.write(f"import auto {os.path.abspath(local_file)}\n")
            f.write("analyze match\n")
            f.write("show statistics\n")
            f.write("quit\n")
            
        res = subprocess.run(f"/usr/games/gnubg -t -q < {cmd_file}", shell=True, capture_output=True, text=True)
        
        pr = None
        for line in res.stdout.split("\n"):
            if "Snowie error rate" in line:
                try:
                    # Line format: Snowie error rate (mili-points/move)      -123.456
                    pr = float(line.split()[-1])
                except:
                    pass
        
        if pr is not None:
            results.append({"file": b.name, "pr": pr})
            print(f"  {b.name}: PR {pr}")
        else:
            print(f"  {b.name}: FAILED TO ANALYZE")
            
    if results:
        avg_pr = sum(r['pr'] for r in results) / len(results)
        print(f"\nBATCH AUDIT COMPLETE")
        print(f"Average Snowie Error Rate: {avg_pr:.3f}")
        
        with open("audit_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
    return results

if __name__ == "__main__":
    batch_audit("expert-eyes-training-742", "logs/")
