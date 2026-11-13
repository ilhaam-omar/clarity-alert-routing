"""Run the structured pipeline on generated, entirely fictional fixtures."""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable,str(ROOT/'scripts/make_demo_data.py')],cwd=ROOT,check=True)
for name in ['01_build_order_timeline.py','02_detect_hazards.py','03_build_longitudinal_states.py','04_identify_action_sets.py','05_build_order_bundles.py','06_apply_dual_capacity.py','07_validate_outputs.py']:
    subprocess.run([sys.executable,str(ROOT/'scripts'/name),'--paths',str(ROOT/'config/synthetic.json')],cwd=ROOT,check=True)
print('Synthetic workflow completed. These outputs are not manuscript results.')
