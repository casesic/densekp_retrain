"""Compare preselected occluder controls using identical GT, boxes and semantics."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.evaluation.candidates import summary


def main():
    directory = Path(sys.argv[1]).resolve()
    report = json.loads((directory / 'report.json').read_text())
    pairs = {}
    for case in report['cases']:
        key = (case['split'], case['occlusion_pair_id'], case['camera'])
        pairs.setdefault(key, {})[case['occlusion_condition']] = case
    results, pooled = [], {}
    for (side, pair, camera), conditions in pairs.items():
        if set(conditions) != {'clear', 'occluded'}:
            raise ValueError(f'Incomplete paired observation: {pair}/{camera}')
        with np.load(conditions['clear']['prediction']) as archive:
            clear = dict(archive)
        with np.load(conditions['occluded']['prediction']) as archive:
            occluded = dict(archive)
        for field in ('target_crop_xy', 'target_erp_xy', 'target_world_xyz',
                      'affine', 'bbox_xyxy', 'supervised', 'in_view'):
            np.testing.assert_array_equal(clear[field], occluded[field])
        supervised = clear['supervised']
        masks = {
            'supervised': supervised,
            'became_occluded': supervised & clear['visible'] & ~occluded['visible'],
            'visible_in_both': supervised & clear['visible'] & occluded['visible'],
            'occluded_in_both': supervised & ~clear['visible'] & ~occluded['visible'],
            'became_visible': supervised & ~clear['visible'] & occluded['visible'],
        }
        metrics = {}
        for name, mask in masks.items():
            metrics[name] = {}
            for field in ('crop_error_px', 'erp_periodic_error_px'):
                a, b = clear[field][mask], occluded[field][mask]
                # Nonfinite predictions remain in the denominator and make the
                # affected aggregate null/failure; never remove them silently.
                with np.errstate(invalid='ignore'):
                    delta = b - a
                metrics[name][field] = dict(clear=summary(a), occluded=summary(b),
                                            occluded_minus_clear=summary(delta))
                group = pooled.setdefault((side, name, field),
                                          dict(clear=[], occluded=[], delta=[]))
                group['clear'].extend(a)
                group['occluded'].extend(b)
                group['delta'].extend(delta)
        results.append(dict(split=side, pair=pair, camera=camera,
                            cases={k: v['id'] for k, v in conditions.items()}, metrics=metrics))
    output = dict(
        status='completed', pairs=results,
        aggregate={'/'.join(key): {k: summary(v) for k, v in values.items()}
                   for key, values in pooled.items()},
        definition='Point-weighted per-semantic endpoint errors; positive delta means worse with occluder.',
        limits='Few preselected controls. Depth-based visibility is approximate; no causal claim about all occluded points.',
    )
    (directory / 'occlusion_control_metrics.json').write_text(
        json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
