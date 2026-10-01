#!/usr/bin/env python3
"""Freeze candidate metadata after a recorded visual review and geometry check."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('assets',type=Path);a=p.parse_args()
    a.assets=a.assets.resolve()
    decision=json.loads((a.assets/'visual_review_decision.json').read_text())
    numeric=json.loads((a.assets/'geometry_acceptance_v2.json').read_text())
    if decision['status']!='passed' or numeric['status']!='passed' or numeric['derivatives']!=8:
        raise ValueError('Both visual and geometry reviews must pass')
    out=a.assets/'review_acceptance.json'
    if out.exists():raise FileExistsError('Already frozen')
    report=json.loads((a.assets/'report.json').read_text())
    review=a.assets/decision['review_directory']
    rendered=json.loads((review/'review_manifest.json').read_text())
    if len(rendered['views'])!=48:raise ValueError('Expected all reference/head and eight-pose views')
    for sheet in decision['reviewed_sheets']:
        if not (review/sheet).is_file():raise FileNotFoundError(sheet)
    appearances={};shapes={}
    for row in report['characters']:
        path=Path(row['manifest']);manifest=json.loads(path.read_text())
        save(path.with_name('manifest_before_freeze.json'),manifest)
        manifest['status']='frozen_after_review'
        manifest['frozen_utc']=datetime.now(timezone.utc).isoformat()
        manifest['visual_review']=str((a.assets/'visual_review_decision.json').resolve())
        manifest['history'].append(dict(version=1,reason='shape accepted unchanged after uniform appearance/geometry review',
                                        material_version=3,clothing_version=2))
        save(path,manifest)
        appearances[row['character_id']]=manifest['appearance']
        shapes[row['character_id']]=dict(model_file=manifest['model_file'],betas=manifest['betas'],proportions=manifest['proportions'])
    save(out,dict(status='passed',asset_root=str(a.assets),split='integration_only',
                  reviewed_utc=datetime.now(timezone.utc).isoformat(),review_directory=str(review),
                  appearances=appearances,shapes=shapes,decision=decision))
    report['status']='characters_frozen_ds_pending';report['pending']=['DS rendering and loader acceptance']
    save(a.assets/'report.json',report)


if __name__=='__main__':main()
