"""Reproduce corrected routing and matched comparisons from authorized frozen inputs."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-input-dir', required=True, type=Path,
                        help='Authorized directory containing the exact frozen Stage 01–03 CSVs')
    parser.add_argument('--extended', action='store_true',
                        help='Also run component, specification, factorial-capacity and historical comparator analyses')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    source = args.frozen_input_dir.expanduser().resolve()
    guard = json.loads((ROOT/'experiments/experiment_source_guard.json').read_text())
    observed = {}
    for name, expected in guard['frozen_inputs'].items():
        path = source/name
        if not path.is_file(): raise FileNotFoundError(path)
        observed[name] = digest(path)
        if observed[name] != expected['sha256']:
            raise ValueError('Frozen input mismatch: '+name+'. Do not bypass the version check.')
    for name, expected in guard['source_files'].items():
        data = (ROOT/name).read_bytes()
        accepted = {expected['sha256'], expected['newline_normalized_sha256'], expected.get('publication_sha256')}
        if hashlib.sha256(data).hexdigest() not in accepted and hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest() not in accepted:
            raise ValueError('Source/configuration mismatch: '+name)
    if args.dry_run:
        print('PASS: exact frozen inputs and scientific source/configuration match. No analysis ran.')
        return
    target = ROOT/'outputs_corrected'
    if target.exists():
        raise FileExistsError('outputs_corrected already exists. Preserve it and use a fresh repository copy.')
    private = target/'private'; private.mkdir(parents=True)
    for name in guard['frozen_inputs']: shutil.copyfile(source/name, private/name)
    config = json.loads((ROOT/'config/mimic_paths.example.json').read_text())
    config['project_root'] = str(ROOT); config['output_dir'] = str(target)
    config_path = ROOT/'config/frozen_reproduction.local.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    stages = ['04_identify_action_sets.py','05_build_order_bundles.py','06_apply_dual_capacity.py','07_validate_outputs.py']
    if args.extended:
        stages += ['09_context_ablation.py','10_specification_sensitivity.py','10a_capacity_factorial_sensitivity.py','10b_static_severity_certainty_comparator.py']
    for name in stages:
        subprocess.run([sys.executable,str(ROOT/'scripts'/name),'--paths',str(config_path)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'experiments/run_matched_comparison.py'),'--project',str(ROOT)],cwd=ROOT,check=True)
    (target/'reproduction_manifest.json').write_text(json.dumps({'frozen_input_hashes':observed,'stages':stages,'extended':args.extended,'python':sys.version},indent=2)+'\n')
    print('Completed. Keep per-record outputs private. Review aggregate exports before sharing.')

if __name__ == '__main__':
    main()
