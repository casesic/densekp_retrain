#!/usr/bin/env python3
"""Bounded full-model updates on the cycling pilot, with fixed-box before/after.

Forest trains, street validates; both reuse the same two people/poses. This is a
pipeline smoke test, not an independent validation split or a selected checkpoint.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.densekp_trainer import DenseKP
from densekp.core.training_audit import CheckUpdates, probes  # Historical re-exports.
from densekp.core.inference import predict_item as predict  # Historical re-export.
from densekp.core.datasets.configuration import audit_cfg
from tools.validate.evaluate_ds_scene_pair import evaluate_scene


def dataset(directory):
    cfg = audit_cfg('supervision_weight')
    return DatasetKeypoints(
        cfg,
        "cycling-smoke",
        is_train=False,
        use_augmentation=False,
        annotation_path=str(directory / "diagnostic_loader.npz"),
        image_root=str(directory),
    )


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'assets_external/densekp.ckpt')
    p.add_argument('--steps',type=int,default=8)
    p.add_argument('--gpu',type=int,default=0)
    args=p.parse_args()
    if args.steps<2: p.error('At least two updates are required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        p.error('This bounded experiment requires CUDA BF16 support')
    args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    args.pilot_root=args.pilot_root.resolve()
    pl.seed_everything(29,workers=True); torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest'); torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    device=torch.device(f'cuda:{args.gpu}')
    names=('cycling_forest','urban_street_assets')
    data={name:dataset(args.pilot_root/name) for name in names}
    with initialize_config_dir(version_base='1.2',config_dir=str(ROOT/'configs/hydra')):
        cfg=compose(config_name='train',overrides=['experiment=densekp'])
    cfg.DATASETS.TARGET_VALUE_SEMANTICS='supervision_weight'
    cfg.TRAIN.BATCH_SIZE=1
    OmegaConf.save(cfg,args.output/'model_config.yaml')
    settings=dict(train_scene=names[0],validation_scene=names[1],steps=args.steps,
                  batch_size=1,precision='bf16-mixed',evaluation_precision='float32',seed=29,
                  augmentation=False,trainable='all backbone and head parameters',
                  source_checkpoint=str(args.checkpoint.resolve()),pilot_root=str(args.pilot_root),
                  split_independent=False,scope='bounded pipeline smoke; repeated identities and poses')
    (args.output/'experiment.json').write_text(json.dumps(settings,indent=2)+'\n')
    checker=CheckUpdates()
    try:
        print('Loading DenseKP for baseline and bounded updates',flush=True)
        model=DenseKP.load_from_checkpoint(str(args.checkpoint),cfg=cfg,map_location='cpu',strict=True).to(device)
        before={name:evaluate_scene(args.pilot_root/name,args.output/'before'/name,model.eval(),device)
                for name in names}
        start=probes(model)
        train_loader=DataLoader(data[names[0]],batch_size=1,shuffle=True,num_workers=0)
        val_loader=DataLoader(data[names[1]],batch_size=1,shuffle=False,num_workers=0)
        trainer=pl.Trainer(accelerator='gpu',devices=[args.gpu],precision='bf16-mixed',
                           max_steps=args.steps,max_epochs=-1,num_sanity_val_steps=0,
                           logger=False,enable_checkpointing=False,enable_progress_bar=False,
                           enable_model_summary=False,callbacks=[checker],default_root_dir=str(args.output))
        trainer.fit(model,train_dataloaders={'img':train_loader},val_dataloaders=val_loader)
        validation=trainer.validate(model,dataloaders=val_loader,verbose=False)[0]
        if validation['val/scored_people_0']!=len(data[names[1]]):
            raise ValueError('Validation did not score every supplied box')
        changes={name:float((value-start[name]).abs().max()) for name,value in probes(model).items()}
        optimizer_steps=[float(state['step']) for state in trainer.optimizers[0].state.values() if 'step' in state]
        if len(checker.attempts)!=args.steps or not optimizer_steps or set(optimizer_steps)!={float(args.steps)}:
            raise ValueError('Actual optimizer steps differ from requested updates')
        if not all(v>0 for v in changes.values()): raise ValueError('A parameter probe did not update')
        # Lightning teardown may move parameters back to CPU.
        model.to(device).eval()
        after={name:evaluate_scene(args.pilot_root/name,args.output/'after'/name,model,device) for name in names}
        expected=predict(model,data[names[1]][0],device)
        checkpoint=args.output/'cycling_smoke_weights.ckpt'
        trainer.save_checkpoint(str(checkpoint),weights_only=True)
        peak=torch.cuda.max_memory_allocated(device)
        del trainer,model
        gc.collect(); torch.cuda.empty_cache()
        restored=DenseKP.load_from_checkpoint(str(checkpoint),cfg=cfg,map_location='cpu',strict=True).to(device)
        actual=predict(restored,data[names[1]][0],device)
        np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)
        np.savez(args.output/'reload_prediction.npz',before_save=expected,after_load=actual)
        result=dict(status='passed',**settings,torch=torch.__version__,lightning=pl.__version__,
                    gpu=torch.cuda.get_device_name(device),actual_optimizer_updates=args.steps,
                    optimizer_attempts=checker.attempts,parameter_probe_changes=changes,
                    validation=validation,peak_cuda_allocated_bytes=peak,
                    reload_max_abs_error=float(np.abs(actual-expected).max()),
                    saved_checkpoint=str(checkpoint),checkpoint_contains_optimizer=False,
                    samples={name:len(value) for name,value in data.items()},
                    metrics={name:dict(before=before[name]['metrics'],after=after[name]['metrics']) for name in names})
        result['quality_status']=('regressed' if any(
            after[name]['metrics']['supervised']['crop_px']['mean'] >
            before[name]['metrics']['supervised']['crop_px']['mean'] for name in names)
            else 'no_regression_on_shared_pose_smoke_only')
        result['checkpoint_use']='diagnostic only; never replaces the baseline'
        (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('status','actual_optimizer_updates','parameter_probe_changes','reload_max_abs_error')}),flush=True)
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps(dict(error=repr(error),attempts=checker.attempts),indent=2)+'\n')
        raise


if __name__=='__main__': main()
