"""Resident Blender worker for editor previews; commands arrive on a private pipe."""
import argparse
import json
import os
from pathlib import Path
import re
import select
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.render import render_multirig_roi


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview-root',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    parent=args.preview_root.resolve()
    print('EDITOR_PREVIEW_WORKER_READY',os.getpid(),flush=True)
    while True:
        readable,_,_=select.select([sys.stdin],[],[],180)
        if not readable:break
        line=sys.stdin.readline()
        if not line:break
        try:
            request=json.loads(line)
            task_id=request['task_id']
            if not isinstance(task_id,str) or re.fullmatch(r'a3_[0-9a-f]{12}',task_id) is None:
                raise ValueError('invalid_preview_task_id')
            run=(parent/task_id/'run').resolve()
            if run!=parent/task_id/'run' or not run.is_dir():
                raise ValueError('invalid_preview_run')
            config=json.loads((run/'run_config.json').read_text())
            if (not config['preview_only'] or config['gpu_uuid']!=args.gpu_uuid or
                len(config['selected_frames'])!=1 or config['max_camera_calls']!=2):
                raise ValueError('invalid_preview_config')
            started=time.monotonic()
            render_multirig_roi.main(run,reuse_session=True)
            spec=json.loads((run/'multirig.json').read_text())
            write(run/'preview_result.json',dict(status='completed',frames=config['selected_frames'],
                rig_ids=[r['rig_id'] for r in spec['rigs']],camera_calls=2,
                sequence_packed=False,resident_worker_pid=os.getpid(),
                worker_seconds=time.monotonic()-started))
            print('EDITOR_PREVIEW_DONE',task_id,flush=True)
        except BaseException as exc:
            try:write(run/'failure.json',dict(stage='resident_preview',error=str(exc),
                                              traceback=traceback.format_exc()))
            except Exception:pass
            traceback.print_exc()
            break


if __name__=='__main__':main()
