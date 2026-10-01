"""Assemble explicit manual reviews; never auto-accept a numerically valid image."""
import json
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1]).resolve()
    destination = root / 'visual_acceptance.json'
    if destination.exists():
        raise FileExistsError('Preserve the completed review record')
    recipe = json.loads((root / 'layout_recipe.json').read_text())
    executions = []
    for phase in ['pilots', 'expand']:
        executions.extend(json.loads((root / f'{phase}_execution.json').read_text())['layouts'])
    if {r['id'] for r in executions} != {r['id'] for r in recipe['layouts']}:
        raise ValueError('Base recipe has unrecorded attempts')
    manual = json.loads((root / 'visual_review_log.json').read_text())
    approved = set(manual['reviewed_accepted_layouts'])
    layouts = []
    for directory in sorted((root / 'layouts').iterdir()):
        if not directory.is_dir():
            continue
        name = directory.name
        side = name.split('_')[0]
        if side not in ['train', 'val']:
            raise ValueError('Unknown layout partition')
        validation = directory / 'validation.json'
        numerical = validation.exists() and json.loads(validation.read_text())['status'] == 'passed'
        if numerical and name not in approved:
            raise ValueError(f'Manual image review missing: {name}')
        if name in approved and not numerical:
            raise ValueError(f'Manual approval conflicts with numerical failure: {name}')
        row = dict(directory=str(directory), partition=side,
                   status='accepted' if numerical else 'rejected')
        if numerical:
            row['excluded_rows'] = [{k: v for k, v in r.items() if k != 'layout'}
                                    for r in manual['excluded_rows'] if r['layout'] == name]
        else:
            log = root / f'{name}.log'
            lines = log.read_text().splitlines() if log.exists() else []
            row['reason'] = next((line for line in reversed(lines) if 'Error:' in line),
                                 'Incomplete or interrupted processing; preserved, not accepted')
            row['failure_log'] = str(log) if log.exists() else None
        if name.endswith('_clear'):
            row['use'] = 'paired_diagnostic_only'
        layouts.append(row)
    if approved - {Path(r['directory']).name for r in layouts}:
        raise ValueError('Reviewed directory is missing')
    output = dict(status='review_complete', layouts=layouts, evidence=manual['evidence'],
        observations=manual['observations'],
        timing='Reviewed and exclusions chosen before any new model predictions',
        policy='ERP and actual full-square/center-input appearance; no zero-intersection clothing gate')
    destination.write_text(json.dumps(output, indent=2) + '\n')
    print('Accepted layouts:', sum(r['status'] == 'accepted' for r in layouts))


if __name__ == '__main__':
    main()
