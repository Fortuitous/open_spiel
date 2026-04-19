import sys
import os
import subprocess

def run_gnubg_analysis(file_path):
    abs_path = os.path.abspath(file_path)
    cmd_file = f"{abs_path}.p"
    with open(cmd_file, "w") as f:
        f.write("new match 1\n")
        f.write("set analysis evaluation 2-ply\n")
        f.write("set dice 64\n")
        f.write("13-7 8-4\n")
        f.write("analyze match\n")
        f.write("show statistics\n")
        f.write("quit\n")

    cmd = f"/usr/games/gnubg -t -q < {cmd_file}"
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout)
    print(result.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smoke_test.py <xg_file>")
        sys.exit(1)
    run_gnubg_analysis(sys.argv[1])
