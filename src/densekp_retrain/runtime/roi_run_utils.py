"""Small shared run file helpers; importing does not start a renderer."""
import json
from pathlib import Path

BLENDER='/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender'


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def disk_snapshot(out):
    total=temporary=0
    for file in Path(out).rglob('*'):
        try:
            if not file.is_file():continue
            size=file.stat().st_size
        except FileNotFoundError:continue
        total+=size
        if '_temporary_ds_source' in file.parts:temporary+=size
    return dict(disk_bytes=total,temporary_bytes=temporary)
