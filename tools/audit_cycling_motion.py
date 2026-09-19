"""Read-only metadata audit; never renders or loads pickled motion objects."""
import csv
import json
import re
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
out = root/'outputs/cycling_motion_audit'
paths = [Path(x) for x in (out/'local_inventory.txt').read_text().splitlines()]
motions, errors, babel_files = [], [], []
for p in paths:
    if p.suffix == '.npz':
        try:
            with np.load(p, allow_pickle=False) as data:
                if 'motion_info' not in data.files:
                    continue
                info = data['motion_info'].tolist()
                motions.append({'motion_path':str(p),'motion_info':info,
                                'frame_count':int(data['poses'].shape[0]),
                                'pose_width':int(data['poses'].shape[-1]),
                                'fps':float(data['mocap_frame_rate']),
                                'gender':str(data['gender'].item())})
        except Exception as exc:
            errors.append({'path':str(p),'error':str(exc)})
    elif p.suffix == '.json' and p.stat().st_size < 30_000_000:
        try:
            text = p.read_text()
            if '"feat_p"' in text and ('"seq_ann"' in text or '"seq_anns"' in text):
                babel_files.append(str(p))
        except (OSError, UnicodeError):
            pass
fields = ['source','babel_label','amass_dataset','amass_motion','bedlam_subject',
          'bedlam_animation_id','gender','frame_count','fps','motion_path','motion_local','confidence']
with (out/'cycling_candidates.csv').open('w',newline='') as stream:
    writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
    for m in motions:
        info = m['motion_info']
        # Require explicit bicycle language; generic ride is insufficient.
        if not re.search(r'\b(ride a bike|cycling|bicycle|biking)\b',str(info[0]),re.I):
            continue
        source = str(info[2]); parts = source.split('/')
        writer.writerow(dict(source='BEDLAM',babel_label=info[0],amass_dataset=parts[-3],
                             amass_motion='/'.join(parts[-3:]),bedlam_subject=Path(m['motion_path']).parent.parent.name,
                             bedlam_animation_id=Path(m['motion_path']).parent.name,gender=m['gender'],
                             frame_count=m['frame_count'],fps=m['fps'],motion_path=m['motion_path'],
                             motion_local='YES',confidence='LABEL_ONLY_REQUIRES_KINEMATICS'))
report = {'inventory_entries':len(paths),'npz_scanned':sum(p.suffix=='.npz' for p in paths),
          'motions':motions,'babel_sequence_annotation_files':babel_files,'npz_errors':errors,
          'scope':'Accessible files in both requested roots; excludes environments, caches, outputs, git and agent state. Archive contents not exhaustive.',
          'public_babel_raw_labels':['ride a bike','ride motorcycle','ride','step ride'],
          'public_labels_source':'https://babel.is.tue.mpg.de/actionsrawlabels.html'}
(out/'local_motion_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='npz_errors'},indent=2))
