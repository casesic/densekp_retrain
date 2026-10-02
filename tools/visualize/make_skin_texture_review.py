#!/usr/bin/env python3
"""Create native-render comparisons using projected SMPL-X joint-weight regions."""


def main():
    import json
    from pathlib import Path
    import numpy as np
    from PIL import Image, ImageDraw

    root=Path(__file__).resolve().parents[2]
    before=root/'outputs/baselines/p0_dressed_textured'; after=root/'outputs/baselines/p0_dressed_skin_textured'
    rows=[json.loads(x) for x in (after/'manifest.jsonl').read_text().splitlines()]
    v=np.load(after/'geometry/vertices.npy')
    model=np.load(Path(json.loads((after/'geometry/geometry_manifest.json').read_text())['model_dir'])/'SMPLX_MALE.npz',allow_pickle=True)
    weights=model['weights']
    regions={'face':[[15,22]], 'hand':[list(range(25,40)),list(range(40,55))], 'foot':[[10],[11]], 'eye':[[23,24]]}
    boxes={}
    for name,groups in regions.items():
        canvas=Image.new('RGB',(640,4*len(groups)*344),(18,18,18)); draw=ImageDraw.Draw(canvas); boxes[name]=[]
        for row_index,(f,joints) in enumerate((f,joints) for f in [0,5,10,15] for joints in groups):
            ids=weights[:,joints].sum(1)>.5
            r=rows[f]; transform=np.linalg.inv(r['camera_matrix_world'])@np.asarray(r['object_matrix_world'])
            q=np.c_[v[f,ids],np.ones(ids.sum())]@transform.T; k=r['intrinsics']
            xy=np.c_[k['fx']*q[:,0]/-q[:,2]+k['cx'],k['cy']-k['fy']*q[:,1]/-q[:,2]]
            lo=np.floor(xy.min(0)-5).astype(int); hi=np.ceil(xy.max(0)+5).astype(int)
            box=(max(0,lo[0]),max(0,lo[1]),min(512,hi[0]),min(512,hi[1])); boxes[name].append({'frame':f,'box':list(map(int,box))})
            for side,directory in enumerate([before,after]):
                crop=Image.open(directory/'rgb'/f'{f:06d}.png').convert('RGB').crop(box)
                crop.thumbnail((320,320)) if max(crop.size)>320 else None
                scale=min(320/crop.width,320/crop.height); crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.NEAREST)
                y=row_index*344; canvas.paste(crop,(side*320+(320-crop.width)//2,y+24+(320-crop.height)//2))
                draw.text((side*320+8,y+5),f'{name} frame {f}: '+('before' if side==0 else 'after'),fill='white')
        canvas.save(after/f'{name}_before_after.png')
    (after/'crop_regions.json').write_text(json.dumps(boxes,indent=2)+'\n')
    canvas=Image.new('RGB',(1024,16*280),(18,18,18)); d=ImageDraw.Draw(canvas)
    for f in range(16):
        for side,directory in enumerate([before,after]):
            im=Image.open(directory/'rgb'/f'{f:06d}.png').convert('RGB'); im.thumbnail((512,256))
            canvas.paste(im,(side*512+128,f*280+24)); d.text((side*512+12,f*280+5),f'Frame {f}: '+('clothing only' if side==0 else 'official skin + clothing; eye blocked'),fill='white')
    canvas.save(after/'contact_sheet_before_after.png')


if __name__ == "__main__":
    main()
