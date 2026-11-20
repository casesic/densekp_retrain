"""A versioned shader-only scalp mask; original face/skin PNG remains unchanged."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def ramp(values,low,high):
    t=np.clip((values-low)/(high-low),0,1)
    return t*t*(3-2*t)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('assets',type=Path)
    parser.add_argument('--version',default='v5')
    args=parser.parse_args();root=args.assets.resolve()
    mp=root/'characters/cycling_F02_v2/manifest.json';manifest=json.loads(mp.read_text());appearance=manifest['appearance']
    reference=np.load(root/'groom_reference/reference.npz');ref=reference['vertices'];x,y,z=ref.T
    mask=np.maximum.reduce([ramp(z,.345,.36),ramp(np.abs(x),.052,.066)*ramp(z,.255,.28)*ramp(y,-.095,-.075),
                            ramp(y,-.005,.015)*ramp(z,.22,.27)])
    mask[z<.20]=0
    mask*=reference['weights'][:,list(reference['group_names']).index('head')]>.1
    folder=mp.parent/f'scalp_shader_{args.version}';folder.mkdir()
    np.save(folder/'vertex_mask.npy',mask.astype(np.float32))
    uv=np.load(appearance['body_uv']);faces=uv['faces'];corners=uv['uv'][uv['faces_uv']]
    forehead=(np.abs(x)<.026)&(z>.333)&(z<.351)&(y<-.065)
    coords=corners[forehead[faces]].copy();alignment=appearance['face_uv_alignment']
    weight=np.ones(len(coords))
    # Match the shader's piecewise-linear UV alignment before sampling this skin.
    for axis,key in [(0,'x_knots'),(1,'y_knots')]:
        a,b,c,d=alignment[key]
        weight*=np.clip((coords[:,axis]-a)/(b-a),0,1)*np.clip((d-coords[:,axis])/(d-c),0,1)
    coords+=weight[:,None]*(coords*np.array([alignment['u_scale']-1,0])+alignment['offset'])
    image=np.asarray(Image.open(appearance['body_texture']).convert('RGB'))/255
    pixels=image[np.clip(((1-coords[:,1])*image.shape[0]).astype(int),0,image.shape[0]-1),
                 np.clip((coords[:,0]*image.shape[1]).astype(int),0,image.shape[1]-1)]
    rgb=np.median(pixels,axis=0);linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    config=dict(vertex_mask=str(folder/'vertex_mask.npy'),color_linear=linear.tolist(),source_skin=appearance['body_texture'],
                source='median of aligned forehead UV samples from the same original skin',
                source_sample_count=len(pixels),source_rgb_srgb=rgb.tolist(),
                method='semantic scalp-only vertex-region blend; no PNG edit',darkness_gate=False,
                limitation='flat scalp fill under groom; visual review required',parent_material_version=appearance['material_version'])
    (folder/'manifest.json').write_text(json.dumps(config,indent=2)+'\n')
    appearance['scalp_replacement']=config;appearance['material_version']=5
    mp.write_text(json.dumps(manifest,indent=2)+'\n')
    report=json.loads((root/'report.json').read_text())
    for row in report['derivatives']:
        if row['character_id']==manifest['character_id']:row['appearance']=appearance
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
