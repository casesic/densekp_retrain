"""Capture independent historical CPU audit behavior, or verify its migration."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
from tests.training_audit_fixture import quiet_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['capture', 'verify'], required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh run directory')
    args = parser.parse_args()
    capture = args.phase == 'capture'
    module_name = ('tools.experiments.train_cycling_smoke' if capture
                   else 'densekp.core.training_audit')
    api = importlib.import_module(module_name)
    if capture and any(symbol.__module__ != module_name for symbol in (api.CheckUpdates, api.probes)):
        raise RuntimeError('Capture requires the independent original, not compatibility exports')
    args.output.mkdir(parents=True, exist_ok=False)
    if capture:
        args.baseline.mkdir(parents=True, exist_ok=False)
    metadata, arrays = quiet_contract(api.CheckUpdates, api.probes, args.output / 'lifecycle')
    (args.output / 'observed.json').write_text(json.dumps(metadata, indent=2) + '\n')
    np.savez(args.output / 'observed.npz', **arrays)
    if capture:
        (args.baseline / 'training_audit_v1.json').write_text(json.dumps(metadata, indent=2) + '\n')
        np.savez(args.baseline / 'training_audit_v1.npz', **arrays)
    else:
        expected = json.loads((args.baseline / 'training_audit_v1.json').read_text())
        assert metadata == expected, 'Lifecycle or exception metadata changed'
        with np.load(args.baseline / 'training_audit_v1.npz', allow_pickle=False) as previous:
            assert set(arrays) == set(previous.files)
            for key, value in arrays.items():
                assert value.shape == previous[key].shape and value.dtype == previous[key].dtype, key
                np.testing.assert_array_equal(value, previous[key], err_msg=key)
    result = dict(status='passed', phase=args.phase, steps=[r['steps'] for r in metadata['runs']],
                  events=len(metadata['events']), arrays=len(arrays), negative_cases=len(metadata['failures']),
                  comparison='exact metadata, tensor values, dtype, shape, parameter order, RNG and modes',
                  network='tiny CPU replacement; real DenseKP loss/AdamW/Lightning/save/resume',
                  densekp_effect_training=False)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
