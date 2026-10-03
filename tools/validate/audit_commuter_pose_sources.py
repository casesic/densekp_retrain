"""Compare existing source/retarget meshes only; never regenerate a pose."""


def main():
    import argparse
    import json
    from pathlib import Path
    import sys
    import numpy as np

    ROOT=Path(__file__).resolve().parents[2];sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
    from densekp_retrain.rendering.blender.intersections import crossings

    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);base=ROOT/'outputs/experiments'
    current=base/'cycling_commuter_pilot_20260930';pose='coco_306800_person_225160'
    ref=current/'characters/cycling_F02_v2/reference/geometry';faces=np.load(ref/'faces.npy');rest=np.load(ref/'vertices.npy')[0]
    mid=rest[faces].mean(1);arm=(np.abs(mid[:,0])>.12)&(np.abs(mid[:,0])<.30)&(mid[:,1]>.28)&(mid[:,1]<.47)
    paths=[('converted_photo_source_P2',base/'cycling_smplx_20260929'/pose/'geometry')]
    for cid in ['cycling_F02_v1','cycling_M02_v1']:
        paths.append((cid,base/'cycling_four_characters_20260929/derivatives'/cid/pose/'geometry'))
    for cid in ['cycling_F02_v2','cycling_M02_v2']:
        paths.append((cid,current/'derivatives'/cid/pose/'geometry'))
    records=[]
    for label,path in paths:
        row=dict(label=label,geometry=str(path))
        if not path.exists():row['status']='path_missing_not_inferred';records.append(row);continue
        f=np.load(path/'faces.npy');row['face_sequence_equal']=bool(np.array_equal(f,faces))
        if not row['face_sequence_equal']:row['status']='region_semantics_unverified';records.append(row);continue
        v=np.load(path/'vertices.npy')[0];pairs,locations,_=crossings(v,f)
        selected=arm[pairs].any(1)
        row.update(proper_crossing_pairs=int(selected.sum()),unique_faces=int(len(np.unique(pairs[selected]))),status='read_only_audit')
        records.append(row)
    a.output.write_text(json.dumps(dict(split='integration_only',region='fixed verified face IDs selected by F02 rest axilla band',
        cases=records,limitation='counts depend on mesh triangulation; they do not measure overlap volume or prove all poses unwearable'),indent=2)+'\n')
    print(json.dumps(records),flush=True)


if __name__ == "__main__":
    main()
