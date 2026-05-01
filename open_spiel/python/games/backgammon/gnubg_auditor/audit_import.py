import subprocess
import sys
import os

def test_import(xg_file):
    abs_path = os.path.abspath(xg_file)
    cmd_file = "import_test.p"
    with open(cmd_file, "w") as f:
        f.write(f"import auto {abs_path}\n")
        f.write("show board\n")
        f.write("analyze match\n")
        f.write("show statistics\n")
        f.write("quit\n")

    cmd = f"/usr/games/gnubg -t -q < {cmd_file}"
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if "Error" in result.stderr or "Illegal" in result.stdout:
        print("IMPORT FAILED!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return False
    else:
        print("IMPORT SUCCESSFUL!")
        # Print the statistics summary
        lines = result.stdout.split("\n")
        for line in lines:
            if "Overall error rate" in line or "Snowie error rate" in line:
                print(line)
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audit_import.py <xg_file>")
        sys.exit(1)
    test_import(sys.argv[1])
