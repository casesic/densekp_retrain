"""Next-turn bounded trial. Defaults to data/config checks; never implicitly trains."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone
from unittest.mock import patch
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp.core.configs import DATASET_FILES,DATASET_FOLDERS
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.densekp_trainer import DenseKP
from densekp.core.crop_geometry import restore_crop_xy
from tools.validate.validate_bedlam_densekp_tiny import audit_cfg
from tools.validate.validate_source_isolated_split import validate_split
from tools.validate.evaluate_candidate_baseline import aggregate
from tools.validate.occlusion_metrics import evaluation_masks, point_errors
from tools.experiments.train_cycling_smoke import CheckUpdates, probes, predict

RUN_OUTPUT = None


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value); self.log.write(value); self.log.flush()

    def flush(self):
        self.stream.flush(); self.log.flush()

    def isatty(self):
        return False


class IndexedRows(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return dict(self.data[index], _frozen_row=index)


class RunAudit(CheckUpdates):
    def __init__(self, out):
        super().__init__(); self.out=out; self.records=[]; self.started=time.monotonic()

    def on_train_start(self, trainer, model):
        optimizer=trainer.optimizers[0]
        expected={id(p) for p in model.get_parameters()}
        actual={id(p) for g in optimizer.param_groups for p in g['params']}
        if not isinstance(optimizer,torch.optim.AdamW) or expected!=actual:
            raise ValueError('Actual optimizer/type/parameter coverage mismatch')
        if model.cfg.MODEL.with_var or model.automatic_optimization or trainer.accumulate_grad_batches!=1:
            raise ValueError('Actual loss/manual optimization/accumulation mismatch')
        if any(not p.requires_grad for p in model.get_parameters()):raise ValueError('Frozen model parameter')
        groups=[]
        for g in optimizer.param_groups:
            if g['lr']!=1e-6 or g['weight_decay']!=1e-4:raise ValueError('Actual LR/weight decay mismatch')
            groups.append(dict(lr=g['lr'],weight_decay=g['weight_decay'],parameters=sum(p.numel() for p in g['params'])))
        if trainer.lr_scheduler_configs:raise ValueError('Unexpected LR scheduler')
        write(self.out/'optimizer_audit.json',dict(status='passed',type=type(optimizer).__name__,groups=groups,
            with_var=model.cfg.MODEL.with_var,manual_optimization=True,accumulation=1,
            precision=str(trainer.precision),all_backbone_head_trainable=True))

    def on_train_batch_start(self,trainer,model,batch,batch_idx):
        if not model.training:raise ValueError('Validation did not restore training mode')
        self.row=int(batch['img']['_frozen_row'].item())
        if batch['img']['img'].shape[0]!=1:raise ValueError('Actual batch size mismatch')

    def on_before_backward(self,trainer,model,loss):
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss before backward')
        self.loss=float(loss.detach())

    def on_before_optimizer_step(self,trainer,model,optimizer):
        super().on_before_optimizer_step(trainer,model,optimizer)
        row=dict(update_attempt=trainer.global_step+1,row_index=self.row,loss=self.loss,
                 seconds_since_start=time.monotonic()-self.started,**self.attempts[-1])
        self.records.append(row)
        with (self.out/'updates.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')

    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        record=dict(completed_update=trainer.global_step,row_index=self.row,loss=self.loss,
            seconds_since_start=time.monotonic()-self.started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(model.device),
            peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved(model.device))
        with (self.out/'completed_updates.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def dataset(root,side):
    train=side=='train'
    with patch.dict(DATASET_FOLDERS,{'source-trial':str(root)}),patch.dict(DATASET_FILES[train],{'source-trial':str(root/'loader'/f'{side}.npz')}):
        return DatasetKeypoints(audit_cfg('supervision_weight'),'source-trial',is_train=train,use_augmentation=False)


class FixedValidation(pl.Callback):
    def __init__(self,data,rows,out,every):
        self.data=data;self.rows=rows;self.out=out;self.every=every;self.reports=[]

    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        step=trainer.global_step
        if not step or step%self.every:return
        folder=self.out/f'step_{step:04d}';folder.mkdir(exist_ok=False)
        states={};cases=[];predictions=[];model.eval();started=time.monotonic()
        versions=[p._version for p in model.parameters()]
        rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state(model.device).clone()
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            for i,row in enumerate(self.rows):
                item=self.data[i]
                raw=model({'img':torch.from_numpy(item['img'][None]).to(model.device)})['pred_keypoints'][0].float().cpu().numpy()
                if raw.shape!=(138,3) or not np.isfinite(raw).all():
                    np.save(folder/f'failed_row_{i}.npy',raw)
                    raise ValueError('Nonfinite/invalid scheduled validation prediction')
                predictions.append(raw);target=item['proj_verts_cropped'][:,:2]
                projected=restore_crop_xy(raw[:,:2],item['_trans'],(256,256))
                gt=self.data.proj_verts[i,:,:2]
                crop,erp=point_errors(raw[:,:2],target,projected,gt,1536)
                invalid=~np.isfinite(raw[:,:2]).all(1);crop[invalid]=np.inf;erp[invalid]=np.inf
                visible=self.data.data['visible'][i];in_view=self.data.data['in_view'][i]
                masks=evaluation_masks(in_view,visible,target);masks['visible_supervised']=masks['supervised']&visible
                cid=str(i);height=row['given_bbox_xyxy'][3]-row['given_bbox_xyxy'][1]
                cases.append(dict(id=cid,action=row['action'],category=row['category'],
                    scale_group='small' if height<96 else ('medium' if height<192 else 'large')))
                states[cid]=dict(crop=crop,erp=erp,masks=masks)
        np.savez(folder/'val_predictions.npz',prediction_raw=predictions)
        if versions!=[p._version for p in model.parameters()] or trainer.global_step!=step:
            raise ValueError('Validation changed parameters or optimizer update count')
        if not torch.equal(rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state(model.device)):
            raise ValueError('Validation changed RNG state')
        report=dict(step=step,metrics=aggregate(cases,states),rows=len(self.rows),
            seconds=time.monotonic()-started,parameters_steps_rng_unchanged=True,
            precision='float32; no autocast',index='unchanged frozen val_index.json',
            keypoint_value='raw third column preserved; never used to select or weight evaluation points')
        write(folder/'validation.json',report);self.reports.append(report)
        trainer.save_checkpoint(str(folder/'checkpoint.ckpt'),weights_only=False)
        model.train()
        print(f'Validation/checkpoint completed: step {step}, rows {len(self.rows)}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--execute-training',action='store_true')
    p.add_argument('--gpu',type=int,default=0);a=p.parse_args();root=a.data_root.resolve()
    recommendation=json.loads((root/'training_recommendation.json').read_text())
    split=json.loads((root/'split_v2.json').read_text())
    indices={s:json.loads((root/f'{s}_index.json').read_text()) for s in ['train','val']}
    isolation=validate_split(split,indices)
    if json.loads((root/'loader/acceptance.json').read_text())['status']!='passed':raise ValueError('Loader not accepted')
    baseline=json.loads((root/'original_baseline/report.json').read_text())
    if baseline['status']!='completed':raise ValueError('Original baseline missing')
    if baseline.get('prediction_failure_cases'):raise ValueError('Original baseline has unresolved nonfinite predictions')
    for label,side in [('train_diagnostic','train'),('val_baseline','val')]:
        if (root/f'{side}_index.json').read_bytes()!=(root/'original_baseline'/f'{label}_frozen_index.json').read_bytes():
            raise ValueError('Input index changed after baseline')
    data={s:dataset(root,s) for s in ['train','val']}
    if any(len(data[s])!=len(indices[s]['rows']) for s in data):raise ValueError('Loader rows differ')
    checkpoint=ROOT/'assets_external/densekp.ckpt';st=checkpoint.stat();old=baseline['checkpoint']
    if (st.st_size,st.st_mtime_ns)!=(old['bytes'],old['mtime_ns']):raise ValueError('Original checkpoint changed')
    settings=dict(data_root=str(root),init_checkpoint=str(checkpoint),checkpoint_bytes=st.st_size,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),logical_gpu=a.gpu,
        checkpoint_mtime_ns=st.st_mtime_ns,strict_load=True,trainable='all backbone/head',
        learning_rate=1e-6,weight_decay=1e-4,batch_size=1,accumulation=1,
        steps=400,precision='bf16-mixed',validation_precision='float32',seed=930,augmentation=False,
        with_var=False,loss='60 * mean supervised squared XY distance; third prediction channel does not weight residuals',
        validation_every=100,sample_exposures=400,equivalent_train_passes=400/len(data['train']),source_isolation=isolation)
    if not a.execute_training:
        print(json.dumps(dict(status='ready_data_config_checks_only',**settings),indent=2));return
    if not a.output:p.error('--output is required for a new training run')
    torch.cuda.set_device(a.gpu)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    global RUN_OUTPUT
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);RUN_OUTPUT=out
    log=(out/'run.log').open('a',buffering=1)
    sys.stdout=Tee(sys.stdout,log);sys.stderr=Tee(sys.stderr,log)
    started=time.monotonic();settings['started_utc']=datetime.now(timezone.utc).isoformat()
    write(out/'settings.json',settings)
    (out/'training_entry.py').write_bytes(Path(__file__).read_bytes())
    (out/'frozen_split_v2.json').write_bytes((root/'split_v2.json').read_bytes())
    for side in ['train','val']:
        (out/f'frozen_{side}_index.json').write_bytes((root/f'{side}_index.json').read_bytes())
    pl.seed_everything(930,workers=True);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    with initialize_config_dir(version_base='1.2',config_dir=str(ROOT/'configs/hydra')):
        cfg=compose(config_name='train',overrides=['experiment=densekp'])
    cfg.DATASETS.TARGET_VALUE_SEMANTICS='supervision_weight';cfg.TRAIN.BATCH_SIZE=1;cfg.TRAIN.LR=settings['learning_rate']
    cfg.TRAIN.WEIGHT_DECAY=settings['weight_decay']
    cfg.MODEL.with_var=False
    cfg.TRAIN.LOSS_TYPE='l2'  # Metadata; keypoint_objective implements squared XY directly.
    OmegaConf.save(cfg,out/'model_config.yaml')
    device=torch.device(f'cuda:{a.gpu}')
    model=DenseKP.load_from_checkpoint(str(checkpoint),cfg=cfg,map_location='cpu',strict=True).to(device)
    # Reuse frozen predictions to check current images/labels/actual merged loaders.
    input_checks=[]
    for side,label in [('train','train_diagnostic'),('val','val_baseline')]:
        cases=[c for c in baseline['cases'] if c['batch']==label]
        for i,case in enumerate(cases):
            item=data[side][i]
            with np.load(case['prediction']) as z:
                np.testing.assert_array_equal(item['img'][:,:,32:-32],z['network_input'])
                np.testing.assert_array_equal(item['proj_verts_cropped'][:,:2],z['target_crop_xy'])
                np.testing.assert_array_equal(item['_trans'],z['affine'])
                np.testing.assert_array_equal(item['proj_verts_cropped'][:,2]>0,z['supervised'])
        input_checks.append(dict(split=side,rows=len(cases),actual_input_gt_affine_mask='exact'))
    fixed_rows=[0,1,2,26,78];precheck=[]
    for i in fixed_rows:
        actual=predict(model,data['val'][i],device)[0]
        with np.load(root/'original_baseline'/f'val_baseline_{i:03d}'/'prediction.npz') as z:expected=z['prediction_raw']
        error=float(np.abs(actual-expected).max());np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)
        precheck.append(dict(val_row=i,max_abs_error=error))
    write(out/'pretraining_baseline_check.json',dict(status='passed',input_checks=input_checks,
        fixed_val_rows=precheck,tolerance=1e-5,precision='FP32',with_var=False))
    torch.cuda.reset_peak_memory_stats(device)
    original=probes(model);checker=RunAudit(out);validation=FixedValidation(data['val'],indices['val']['rows'],out,100)
    trainer=pl.Trainer(accelerator='gpu',devices=[a.gpu],precision='bf16-mixed',max_steps=400,max_epochs=-1,
        num_sanity_val_steps=0,limit_val_batches=0,logger=False,enable_checkpointing=False,
        enable_progress_bar=False,enable_model_summary=False,callbacks=[checker,validation],default_root_dir=str(out))
    fit_start=time.monotonic()
    trainer.fit(model,train_dataloaders={'img':DataLoader(IndexedRows(data['train']),batch_size=1,shuffle=True,num_workers=0)})
    fit_seconds=time.monotonic()-fit_start
    steps={float(s['step']) for s in trainer.optimizers[0].state.values() if 'step' in s}
    if len(checker.attempts)!=400 or steps!={400.}:raise ValueError('Optimizer update count mismatch')
    changes={k:float((v-original[k]).abs().max()) for k,v in probes(model).items()}
    if not all(v>0 for v in changes.values()):raise ValueError('Parameter group did not update')
    if [r['step'] for r in validation.reports]!=[100,200,300,400]:raise ValueError('Scheduled validations missing')
    model.to(device);expected=np.concatenate([predict(model,data['val'][i],device) for i in fixed_rows]);peak=torch.cuda.max_memory_allocated(device)
    reserved=torch.cuda.max_memory_reserved(device)
    del trainer,model;gc.collect();torch.cuda.empty_cache()
    restored=DenseKP.load_from_checkpoint(str(out/'step_0400/checkpoint.ckpt'),cfg=cfg,map_location='cpu',strict=True).to(device)
    actual=np.concatenate([predict(restored,data['val'][i],device) for i in fixed_rows]);np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=0)
    np.savez(out/'reload_predictions.npz',before_save=expected,after_load=actual,val_rows=fixed_rows)
    if (checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)!=(st.st_size,st.st_mtime_ns):raise ValueError('Original checkpoint changed')
    write(out/'report.json',dict(status='workflow_passed_not_automatic_effect_pass',settings=settings,
        actual_updates=400,parameter_changes=changes,reload_max_abs_error=float(np.abs(actual-expected).max()),
        peak_cuda_bytes=peak,peak_reserved_cuda_bytes=reserved,fit_seconds=fit_seconds,total_seconds=time.monotonic()-started,
        finished_utc=datetime.now(timezone.utc).isoformat(),sample_exposures=len(checker.records),
        unique_exposed_rows=len({r['row_index'] for r in checker.records}),
        loss_range=[min(r['loss'] for r in checker.records),max(r['loss'] for r in checker.records)],
        validation_steps=[r['step'] for r in validation.reports],
        comparison='Use frozen original baseline; primary predeclared final step 400, report all scheduled points and complete-control regressions'))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        if RUN_OUTPUT is not None:
            write(RUN_OUTPUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
                utc=datetime.now(timezone.utc).isoformat(),policy='stopped; no sample skipping or automatic retry'))
        raise
