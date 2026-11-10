"""Read at most six rows through A1's Dataset without importing its model stack."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import types

import numpy as np


def load_reader(repo):
    repo=Path(repo).resolve();landmarks=repo/'landmarks';lib=landmarks/'lib';datasets=lib/'datasets'
    for name,path in [('landmarks',landmarks),('landmarks.lib',lib),('landmarks.lib.datasets',datasets)]:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    return importlib.import_module('landmarks.lib.datasets.mamma_roi_pilot').MammaManifestDataset


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--reader-repo',type=Path,default=Path('/mnt/data1/yy/home/yy/Projects/mamma_retrain'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit',type=int,default=6)
    args=parser.parse_args()
    if not 1<=args.limit<=6:raise ValueError('readback_limit_must_be_1_to_6')
    cls=load_reader(args.reader_repo);began=time.monotonic();dataset=cls(args.config);rows=[]
    for index in range(min(args.limit,len(dataset))):
        item=dataset[index]
        rows.append(dict(index=index,identity=item['identity'],image_shape=list(item['img'].shape),xy_shape=list(item['xy_erp'].shape),
            coordinate_valid=int(item['coordinate_loss_valid'].sum()),visibility_known=int((item['visibility']>=0).sum()),
            visibility_loss_valid=int(item['visibility_loss_valid'].sum()),source_sampling_supported=bool(item['source_sampling_supported']),
            finite=bool(np.isfinite(item['xy_erp']).all() and np.isfinite(item['xy_model']).all())))
    result=dict(schema='mamma_a1_reader_readback_v1',reader='read_only_mamma_retrain.MammaManifestDataset',
        selected_ready_indices=list(range(len(rows))),read_count=len(rows),total_observed=len(dataset),seconds=time.monotonic()-began,rows=rows,status='passed')
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
