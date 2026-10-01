"""Measured mesh sections and component-level attribution, without rendering."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def section(vertices,faces,axis,value):
    triangles=vertices[faces]
    selected=(triangles[:,:,axis].min(1)<value)&(triangles[:,:,axis].max(1)>value)
    triangles=triangles[selected];segments=[]
    for triangle in triangles:
        points=[]
        for a,b in zip(triangle,np.roll(triangle,-1,axis=0)):
            if (a[axis]>value)!=(b[axis]>value):
                points.append(a+(b-a)*(value-a[axis])/(b[axis]-a[axis]))
        if len(points)==2:segments.append(points)
    return np.asarray(segments).reshape(-1,2,3)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('diagnosis',type=Path);args=p.parse_args()
    report=json.loads((args.diagnosis/'diagnosis.json').read_text());out=[]
    for c in report['cases']:
        folder=args.diagnosis/c['character_id']/c['pose'];z=np.load(folder/'evidence.npz')
        meta=json.loads((Path(c['clothing'])/'clothing_manifest.json').read_text())
        components=[]
        for component in meta['components']:
            start,n=component['vertex_start'],component['vertex_count']
            face_ids=np.flatnonzero(((z['cloth_faces']>=start)&(z['cloth_faces']<start+n)).all(1))
            entry={'name':component['name']}
            for stage in ['bound','final']:
                pairs=z[stage+'_self_pairs'];own=np.isin(pairs,face_ids)
                entry[stage+'_body_crossings']=int(np.isin(z[stage+'_body_pairs'][:,0],face_ids).sum())
                entry[stage+'_internal_crossings']=int(own.all(1).sum())
                entry[stage+'_other_component_crossings']=int((own.any(1)&~own.all(1)).sum())
            components.append(entry)
        out.append({'character':c['character_id'],'pose':c['pose'],'components':components})
        if c['character_id']!='cycling_F02_v2':continue
        # Reference sagittal sections expose the chest/abdomen transition;
        # P2 planes sample more than the single old winding-number location.
        values=[0,.05,.10] if c['pose']=='reference' else [.02,.04,.06]
        fig,axes=plt.subplots(1,3,figsize=(12,5),layout='constrained')
        topcount=meta['components'][0]['vertex_count'];cf=z['cloth_faces'];cf=cf[(cf<topcount).all(1)]
        for ax,x in zip(axes,values):
            for key,faces,color in [('body',z['body_faces'],'0.3'),('bound',cf,'#2166ac'),('final',cf,'#d6604d')]:
                lines=section(z[key],faces,0,x)
                ax.add_collection(LineCollection(lines[:,:,[2,1]],colors=color,linewidths=1,label=key))
            ax.set_xlim(-.2,.38);ax.set_ylim(.05,.57);ax.set_aspect('equal');ax.set_title(f'x = {x:.3f} m')
            ax.set_xlabel('z / m');ax.set_ylabel('y / m');ax.grid(alpha=.2)
        axes[0].legend();fig.suptitle(f"F02 {c['pose']}: actual triangle-plane sections")
        fig.savefig(folder/'sections.png',dpi=160);plt.close(fig)
    (args.diagnosis/'component_attribution.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
