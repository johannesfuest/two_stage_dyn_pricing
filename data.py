import subprocess
import sys

scripts = [
    "TRACE_data.py",
    "LSEG_data.py",
    "preprocessing_data.py"
]

for script in scripts:
    print(f"\nRunning {script}...")
    try:
        # Run each script and inherit stdout/stderr so you can see the error
        result = subprocess.run(
            [sys.executable, script],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Script {script} failed with exit code {e.returncode}")
        sys.exit(1)

print("\n✅ All scripts ran successfully.")
