"""CPU-only preservation audit after an interrupted trial; never resumes training."""
import argparse
import gc
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True)
    a=p.parse_args();root=a.trial.resolve();torch.set_num_threads(4)
    settings=json.loads((root/'settings.json').read_text())
    original=torch.load(settings['init_checkpoint'],map_location='cpu')['state_dict']
    keys={group:next(k for k in original if k.startswith(group+'.')) for group in ['backbone','head']}
    probes={g:original[k].flatten()[:4096].clone() for g,k in keys.items()}
    del original;gc.collect();records=[]
    for file in sorted(root.glob('step_*/checkpoint.ckpt')):
        payload=torch.load(file,map_location='cpu');state=payload['state_dict']
        if not all(bool(torch.isfinite(t).all()) for t in state.values()):raise ValueError('Nonfinite checkpoint state')
        optimizer=payload['optimizer_states'][0]
        steps=sorted({float(s['step']) for s in optimizer['state'].values() if 'step' in s})
        cfg=payload['hyper_parameters']['cfg']
        validation=json.loads((file.parent/'validation.json').read_text())
        with np.load(file.parent/'val_predictions.npz') as z:
            assert z['prediction_raw'].shape==(79,138,3) and np.isfinite(z['prediction_raw']).all()
        records.append(dict(path=str(file),bytes=file.stat().st_size,mtime_ns=file.stat().st_mtime_ns,
            global_step=payload['global_step'],optimizer_steps=steps,state_keys=len(state),all_state_finite=True,
            with_var=bool(cfg.MODEL.with_var),optimizer_groups=[dict(lr=g['lr'],weight_decay=g['weight_decay']) for g in optimizer['param_groups']],
            parameter_probe_changes={g:float((state[k].flatten()[:4096]-probes[g]).abs().max()) for g,k in keys.items()},
            validation_rows=validation['rows'],diagnostic_supervised_mean=validation['metrics']['all/supervised']['point_weighted']['crop_256_px']['mean']))
        assert steps==[float(payload['global_step'])]
        del payload,state,optimizer;gc.collect()
    updates=[json.loads(line) for line in (root/'updates.jsonl').read_text().splitlines()]
    stat=Path(settings['init_checkpoint']).stat()
    assert (stat.st_size,stat.st_mtime_ns)==(settings['checkpoint_bytes'],settings['checkpoint_mtime_ns'])
    result=dict(status='saved_partial_checkpoints_valid_not_completed_trial',utc=datetime.now(timezone.utc).isoformat(),
        checkpoints=records,recorded_update_attempts=len(updates),confirmed_completed_updates_lower_bound=updates[-1]['step'],
        last_attempt_completion='unknown; record is before optimizer.step',sample_exposures=len(updates),
        unique_exposed_rows=len({r['row_index'] for r in updates}),
        finite_loss_range=[min(r['loss'] for r in updates),max(r['loss'] for r in updates)],
        recorded_seconds_to_last_attempt=updates[-1]['seconds_since_start'],
        original_checkpoint_stats_unchanged=True,workflow_passed=False,effect_passed=None,
        missing=['300/400 checkpoints','400-step strict reload','400-step train/val/paired/old51 evaluation'],
        peak_cuda_bytes=None,peak_memory_note='GPU failed before end-of-run memory counters could be queried')
    (root/'interruption_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
