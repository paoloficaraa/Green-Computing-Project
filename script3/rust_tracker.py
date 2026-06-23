import subprocess
import os
import sys
from codecarbon import EmissionsTracker

root_dir = os.path.dirname(os.path.abspath(__file__))
reports_dir = os.path.join(root_dir, "../CodeCarbon reports")
os.makedirs(reports_dir, exist_ok=True)

is_windows = sys.platform.startswith('win')
exe_name = "target/release/script3.exe" if is_windows else "target/release/script3"
binary_path = os.path.join(root_dir, exe_name)

if not os.path.isfile(binary_path):
    print("Errore: Eseguibile non trovato. Esegui 'cargo build --release' prima di avviare il tracker.")
    sys.exit(1)

tracker = EmissionsTracker(
    output_file=os.path.join(reports_dir, "emissions_script3.csv"),
    measure_power_secs=1,
    tracking_mode="process",
    log_level="error"
)

print("Avvio misurazione...")
tracker.start()

try:
    subprocess.run([binary_path], check=True)
except Exception as e:
    print(f"Errore: {e}")
finally:
    tracker.stop()
    print("Misurazione completata.")