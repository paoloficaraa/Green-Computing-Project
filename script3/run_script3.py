import subprocess
import sys
from pathlib import Path
from codecarbon import EmissionsTracker

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPORTS_DIR = PROJECT_ROOT / "CodeCarbon reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

output_file = REPORTS_DIR / "emissions_script3.csv"

print("Starting CodeCarbon tracking...")
tracker = EmissionsTracker(output_file=str(output_file))
tracker.start()

try:
    print(f"Launching native Julia script...")
    result = subprocess.run(
        ["julia", "-t", "auto", "--project=" + str(SCRIPT_DIR), str(SCRIPT_DIR / "script3.jl")],
        cwd=str(PROJECT_ROOT),
        check=True
    )
    print("Julia execution completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error during Julia execution: {e}", file=sys.stderr)
    sys.exit(1)
finally:
    print("Stopping CodeCarbon tracking...")
    tracker.stop()
    print(f"Emissions report saved to: {output_file}")
