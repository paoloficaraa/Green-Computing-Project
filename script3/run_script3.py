import subprocess
import sys
from pathlib import Path
from codecarbon import EmissionsTracker

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPORTS_DIR = PROJECT_ROOT / "CodeCarbon reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

output_file = REPORTS_DIR / "emissions_script3.csv"

# CodeCarbon appende al file esistente: rimuoviamo il vecchio report per avere una misurazione pulita di questa esecuzione
if output_file.exists():
    output_file.unlink()
    print(f"Removed previous emissions report: {output_file}")

print("Starting CodeCarbon tracking...")
# Usiamo l'EmissionsTracker di CodeCarbon per monitorare l'intero sistema durante l'esecuzione dello script Julia nativo
tracker = EmissionsTracker(output_file=str(output_file))
tracker.start()

try:
    print(f"Launching native Julia script as a subprocess (using all available threads)...")
    # Lanciamo Julia con --project puntando a script3/ (dove c'è il Project.toml nativo)
    # e abilitando i thread con '-t', 'auto' per usare tutti i core
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
