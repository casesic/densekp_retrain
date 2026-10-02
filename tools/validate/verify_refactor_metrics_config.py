"""Verify preserved pre-migration CPU evidence using current public modules.

Capture was retired in pruning B1. It is rejected before optional imports or
file access. Verify reads the original seven-sample baseline and saved predictions;
no model is constructed and baseline files are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]



def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def fixture_inputs():
    """Adversarial cases independent of model/data availability; preserve dtype."""
    cases = [dict(id='a', action='walking', category='control', scale_group='small',
                  lens_truncated_gt=False, neighbor_crop_pixels=0, in_view_outside_full_crop=0),
             dict(id='b', action='cycling', category='edge', scale_group='large',
                  lens_truncated_gt=True, neighbor_crop_pixels=3, in_view_outside_full_crop=1),
             dict(id='c', action='walking', category='control', scale_group='small')]
    states = {
        'a': dict(crop=np.array([1., 1., 99.], np.float32),
                  erp=np.array([2., 2., 99.], np.float64),
                  masks=dict(supervised=np.array([1, 1, 0], bool), empty=np.zeros(3, bool))),
        'b': dict(crop=np.array([10., 100., np.inf], np.float32),
                  erp=np.array([20., 200., np.inf], np.float64),
                  masks=dict(supervised=np.array([1, 0, 0], bool), empty=np.zeros(3, bool))),
        'c': dict(crop=np.array([np.nan], np.float32), erp=np.array([np.inf]),
                  masks=dict(supervised=np.zeros(1, bool), empty=np.zeros(1, bool))),
    }
    fixtures = []
    for name in ('summarize', 'summary'):
        for label, values in [('empty', []), ('finite', [1., 2., 10.]),
                              ('nan', [2., np.nan, 4.]), ('inf', [2., np.inf, 4.]),
                              ('negative_inf', [-np.inf])]:
            fixtures.append((name + '_' + label, name, [np.asarray(values, np.float32)]))
    fixtures.extend([
        ('aggregate_unequal_and_empty', 'aggregate', [cases, states]),
        ('aggregate_no_cases', 'aggregate', [[], {}]),
        ('crop_bounds', 'evaluation_masks', [np.array([1, 1, 1, 1, 0, 1, 1], bool),
            np.array([1, 0, 1, 0, 1, 0, 1], bool),
            np.array([[-.5, 0], [.5, 0], [-.375, 0], [.375, 0], [0, 0], [np.nan, 0], [0, -.5]])]),
        ('pair_groups', 'pair_masks', [np.array(x, bool) for x in
            [[1, 1, 1, 1, 0], [1, 1, 0, 0, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1]]]),
        ('rejected_still_counted', 'prediction_retention', [np.array([1, 1, 0], bool), np.array([1, 0, 1], bool)]),
        ('retention_empty', 'prediction_retention', [np.zeros(3, bool), np.ones(3, bool)]),
    ])
    for width in (1536, 5760):
        fixtures.append((f'periodic_{width}', 'point_errors', [np.array([[0., 0.], [.125, 0.]], np.float32),
            np.zeros((2, 2), np.float32), np.array([[width-1., 10.], [1., 10.]]),
            np.array([[1., 10.], [width-1., 10.]]), width]))
    failing_states = {key: dict(value) for key, value in states.items()}
    failing_states['b'] = dict(states['b'], crop=np.array([np.inf, 100., 0.], np.float32))
    fixtures.append(('aggregate_failure_denominator', 'aggregate', [cases, failing_states]))
    return fixtures


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(array=value.tolist(), dtype=str(value.dtype), shape=list(value.shape))
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def json_safe(value):
    """Encode nonfinite fixture inputs explicitly, never as nonstandard JSON."""
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def functions():
    from densekp.core import evaluation as metrics
    from densekp.core.datasets import configuration as config
    from densekp_retrain.evaluation import candidates

    result = {name: getattr(metrics, name) for name in
              ['summarize', 'point_errors', 'evaluation_masks', 'prediction_retention', 'pair_masks']}
    result.update(summary=candidates.summary, aggregate=candidates.aggregate, audit_cfg=config.audit_cfg)
    return result


def exercise_fixtures(api):
    records = []
    for label, name, arguments in fixture_inputs():
        try:
            outcome = dict(value=encode(api[name](*arguments)))
        except ValueError as error:
            outcome = dict(error_type=type(error).__name__, message=str(error))
        records.append(json_safe(dict(id=label, function=name, arguments=encode(arguments), outcome=outcome)))
    return records


def compare_array(actual, expected):
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(f'Array contract changed: {actual.dtype}/{actual.shape} vs {expected.dtype}/{expected.shape}')
    np.testing.assert_array_equal(actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['capture', 'verify'], required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--report', type=Path, help='New verification report path; never overwritten')
    parser.add_argument('--selection', type=Path,
                        default=ROOT / 'outputs/plans/full_refactor_20261001/selected_baselines.json')
    args = parser.parse_args()
    if args.phase == 'capture':
        parser.error('capture retired in B1; use preserved pre-migration baselines with --phase verify')

    # Defer data dependencies until after the retired mode has been rejected.
    global np
    import numpy as np
    from densekp.core.crop_geometry import restore_crop_xy
    from densekp.core.datasets.dataset_keypoints import DatasetKeypoints

    api = functions()
    baseline = args.baseline.resolve()
    if not args.report or args.report.exists():
        parser.error('--report must be a new file for verification')
    selection = json.loads((baseline / 'selection.json').read_text())
    for value, name in [(None, 'undeclared_loader.yaml'), ('supervision_weight', 'loader.yaml')]:
        dumped = api['audit_cfg'](value).dump()
        assert dumped == (baseline / name).read_text(), 'Configuration values changed'
    fixtures = exercise_fixtures(api)
    assert fixtures == json.loads((baseline / 'metric_fixtures.json').read_text()), 'Metric fixture changed'
    cases, states, records = [], {}, []
    reports = {}
    for entry in selection['samples']:
        case_id = entry['id']
        source_report = Path(entry['prediction']).parent.parent / 'report.json'
        if source_report not in reports:
            reports[source_report] = {case['id']: case for case in json.loads(source_report.read_text())['cases']}
        case = reports[source_report][case_id]
        dataset = DatasetKeypoints(
            api["audit_cfg"]("supervision_weight"),
            "refactor-readback",
            is_train=False,
            use_augmentation=False,
            annotation_path=entry["loader_input"],
            image_root=entry["scene"],
        )
        item = dataset[entry['loader_row']]
        with np.load(entry['prediction'], allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        arrays = {key: np.asarray(value) for key, value in item.items()}
        arrays['network_input'] = item['img'][:, :, 32:-32]
        arrays['prediction_raw'] = saved['prediction_raw']
        arrays['in_view'] = saved['in_view']
        arrays['target_erp_xy'] = saved['target_erp_xy']
        masks = api['evaluation_masks'](saved['in_view'], saved['visible'], item['proj_verts_cropped'][:, :2])
        masks['visible_supervised'] = masks['supervised'] & saved['visible']
        pred_erp = restore_crop_xy(saved['prediction_raw'][:, :2], item['_trans'], (256, 256))
        crop, erp = api['point_errors'](saved['prediction_raw'][:, :2], item['proj_verts_cropped'][:, :2],
                                        pred_erp, saved['target_erp_xy'], entry['image_size_wh'][0])
        arrays.update(prediction_erp_xy=pred_erp, crop_error_px=crop, erp_periodic_error_px=erp)
        arrays.update({'mask_' + name: value for name, value in masks.items()})
        compare_array(arrays['network_input'], saved['network_input'])
        compare_array(item['_trans'], saved['affine'])
        compare_array(item['proj_verts_cropped'][:, :2], saved['target_crop_xy'])
        compare_array(masks['supervised'], saved['supervised'])
        frozen = baseline / (case_id + '.npz')
        with np.load(frozen, allow_pickle=False) as previous:
            assert set(previous.files) == set(arrays)
            for name, array in arrays.items():
                compare_array(array, previous[name])
        cases.append(case)
        states[case_id] = dict(crop=crop, erp=erp, masks=masks)
        records.append(dict(id=case_id, arrays=len(arrays), input_shape=list(item['img'].shape),
                            input_dtype=str(item['img'].dtype), supervised=int(masks['supervised'].sum())))
    aggregate = api['aggregate'](cases, states)
    assert cases == json.loads((baseline / 'cases.json').read_text()), 'Case metadata changed'
    # Serialized text also freezes dictionary insertion order and null fields.
    expected = json.loads((baseline / 'aggregate.json').read_text())
    assert json.dumps(aggregate, allow_nan=False) == json.dumps(expected, allow_nan=False)
    result = dict(status='passed', phase=args.phase, utc=datetime.now(timezone.utc).isoformat(),
                  baseline=str(baseline), samples=records, metric_fixtures=len(fixtures),
                  aggregate_groups=len(aggregate), comparison='exact values, shapes, dtypes, configuration dumps and metric outcomes',
                  model_inference=False, training=False, rendering=False,
                  head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
