"""Render only bounded, already-probed placement repairs into new directories."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--layouts', nargs='+', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / 'ground_render_execution.json'
    records = json.loads(path.read_text())['layouts'] if path.exists() else []
    for name in args.layouts:
        repair = json.loads((root / 'ground_repairs' / name / 'repair.json').read_text())
        if any(r['id'] == repair['id'] for r in records):
            continue
        if repair['status'] != 'ground_probe_passed_render_pending':
            raise ValueError('Native ground probe must pass first')
        output = root / 'layouts' / repair['id']
        command = [sys.executable, str(ROOT / 'tools/experiments/render_adult_candidates_ds.py'),
            '--assets', str(root / 'assets'), '--output', str(output),
            '--characters', *repair['characters'], '--pose-source-ids', *repair['pose_source_ids'],
            '--scene', repair['scene'], '--reuse-layout', repair['reuse_layout'],
            '--width', '1536', '--samples', '24', '--device', 'CUDA',
            '--output-mode', 'full', '--final-width', '1536', '--final-height', '768']
        started = time.monotonic()
        utc = datetime.now(timezone.utc).isoformat()
        print('START', repair['id'], flush=True)
        with (root / (repair['id'] + '.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        records.append(dict(id=repair['id'], parent_layout=name, partition=repair['partition'],
            directory=str(output), returncode=result.returncode, seconds=time.monotonic()-started,
            started_utc=utc, finished_utc=datetime.now(timezone.utc).isoformat(), command=command,
            status='numerical_complete_visual_pending' if not result.returncode else 'failed_preserved'))
        path.write_text(json.dumps(dict(layouts=records), indent=2) + '\n')
        print('DONE', repair['id'], result.returncode, flush=True)


if __name__ == '__main__':
    main()
