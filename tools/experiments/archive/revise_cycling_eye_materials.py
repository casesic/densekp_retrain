#!/usr/bin/env python3
"""Version the four-character eye material correction; preserve all geometry."""
import argparse
import copy
import json
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('assets',type=Path)
    p.add_argument('--version',type=int,choices=[2,3],default=2)
    a=p.parse_args(); report_path=a.assets/'report.json'
    report=json.loads(report_path.read_text())
    version=a.version
    backup=a.assets/f'report_before_material_v{version}.json'
    if backup.exists(): raise FileExistsError('Material revision already exists')
    save(backup,report)
    appearances={}
    for row in report['characters']:
        path=Path(row['manifest']); manifest=json.loads(path.read_text())
        save(path.with_name(f'manifest_before_material_v{version}.json'),manifest)
        old=copy.deepcopy(manifest['appearance'])
        appearance=manifest['appearance']; appearance['material_version']=version
        appearance['eye_patch']['centers']=[]
        if version==2:
            appearance['eye_uv_warps']=[
                dict(center=[.252,.797],radii=[.060,.055],offset=[-.013,.024]),
                dict(center=[.351,.797],radii=[.060,.055],offset=[.003,.024])]
        else:
            appearance['eye_uv_warps']=[]
            appearance['face_uv_alignment']=dict(x_knots=[.12,.18,.42,.48],y_knots=[.74,.79,.85,.94],
                                                  u_scale=1.15,offset=[-.048,.024])
        history=manifest.get('material_history',[])
        history.append(dict(version=version-1,appearance=old,status='rejected',reason=(
            'forehead sampling leaves misplaced eyebrows' if version==2 else 'elliptical displacement folds UV around eyebrows')))
        history.append(dict(version=version,status='candidate',reason=(
            'local UV alignment' if version==2 else 'monotonic broad-band upper-face UV alignment; body and topology unchanged')))
        manifest['material_history']=history
        save(path,manifest); appearances[row['character_id']]=appearance
    for derivative in report['derivatives']:
        derivative['appearance']=appearances[derivative['character_id']]
    save(report_path,report)
    for path in a.assets.glob('**/geometry/geometry_manifest.json'):
        manifest=json.loads(path.read_text())
        save(path.with_name(f'geometry_manifest_before_material_v{version}.json'),manifest)
        manifest['appearance']=appearances[manifest['character_id']]
        save(path,manifest)
    save(a.assets/f'material_revision_v{version}.json',dict(split='integration_only',geometry_changed=False,
         source_textures_changed=False,status='candidate_pending_head_review',appearances=appearances))


if __name__=='__main__': main()
