"""CPU-only physical-domain projection review; no RGB render or model."""
import argparse,json,sys,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import numpy as np
import cv2
from PIL import Image
from densekp_retrain.data.dynamic_scene import geometry_paths,person_matrices
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.camera.erp import project_erp

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episode',type=Path,required=True)
    ap.add_argument('--frames',help='Comma-separated CPU projection review frames; default legacy first/middle/last/edge-end')
    a=ap.parse_args();root=a.episode
    out=root/'projection_review';out.mkdir(exist_ok=False)
    recipe=json.loads((root/'recipe.json').read_text());stats=json.loads((root/'selected_projection.json').read_text());regions=json.loads((root/'planned_regions.json').read_text());world=np.load(root/'world_densekp.npy')
    poses={c['id']:np.array(c['camera_to_world']) for c in json.loads((root/'rig.json').read_text())['cameras']};masks={c:np.asarray(Image.open(root/'calibration'/f'{c}_valid_mask.png'))>0 for c in poses}
    parts=[]
    for p in recipe['people']:
        asset=json.loads(Path(p['asset_manifest']).read_text());paths=geometry_paths(asset)
        parts.append((np.load(paths['body_vertices'],mmap_mode='r'),np.load(paths['body_faces']),np.load(paths['cloth_vertices'],mmap_mode='r'),np.load(paths['cloth_faces']),person_matrices(p,recipe['frame_count'])))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    selected=([int(value) for value in a.frames.split(',')] if a.frames else
              sorted({0,recipe['frame_count']//2,recipe['frame_count']-1,stats['dual_edge']['windows'][0]['last']}))
    if len(set(selected))!=len(selected) or any(k<0 or k>=recipe['frame_count'] for k in selected):
        raise ValueError('invalid_projection_review_frames')
    for k in selected:
        fig,axes=plt.subplots(1,2,figsize=(16,7))
        for ax,(c,pose) in zip(axes,poses.items()):
            r=regions[k][c]['roi_xyxy'];x0,y0,x1,y1=r
            ax.imshow(masks[c],cmap='gray',vmin=-.3,vmax=1.4,extent=(0,5760,2880,0),interpolation='nearest')
            contours,_=cv2.findContours(masks[c].astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                q=contour[:,0];ax.plot(q[:,0],q[:,1],color='#ffd000',lw=1.5)
            for j,(body,faces,cloth,cf,mats) in enumerate(parts):
                p=recipe['people'][j];ix=p['asset_frames'][k];M=mats[k];color=['#2a85ff','#ab64cd'][j]
                silhouette=np.zeros((720,1440),np.uint8)
                for vv,ff in [(body,faces),(cloth,cf)]:
                    xy=project_erp(vv[ix]@M[:3,:3].T+M[:3,3],pose,5760,2880)[0]
                    cv2.fillPoly(silhouette,np.rint(xy[ff]/4).astype(np.int32),1)
                for contour in cv2.findContours(silhouette,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
                    q=contour[:,0]*4;ax.fill(q[:,0],q[:,1],color=color,alpha=.2);ax.plot(q[:,0],q[:,1],color=color,lw=.6)
                xy,_,_,_,inside=point_in_view(world[k,j],pose,c,masks[c]);ax.scatter(*xy[inside].T,s=6,c=color);ax.scatter(*xy[~inside].T,s=7,c='red',marker='x')
                ax.text(*xy.mean(0),f'p{p["person_id"]}',color=color,weight='bold')
            row=stats['frames'][k]['views'][c];cx,cy=row['crop_center'];side=row['crop_scale']*200
            ax.add_patch(Rectangle((cx-side/2,cy-side/2),side,side,fill=False,color='limegreen',lw=1.2,label='target projected full square'))
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='cyan',linestyle='--',label='planned rendered ROI'))
            ax.set_xlim(x0-40,x1+40);ax.set_ylim(y1+40,y0-40);ax.set_aspect('equal');ax.set_title(f'{c}: p1 inside/outside={row["in_view"]}/{row["outside_domain"]}; body inside={row["body_domain_fraction"]:.2f}')
            ax.set_xlabel('Full ERP u (px)');ax.set_ylabel('Full ERP v (px)');ax.legend(loc='lower left',fontsize=8)
        fig.suptitle(f'CPU projected body+cloth contours; t={k/10:.1f}s | yellow physical domain; red out-of-domain R138; not rendered RGB\ncommon domain={stats["frames"][k]["common_physical_points"]}; projected crop common={stats["frames"][k]["common_projected_crop_points"]}')
        fig.tight_layout();fig.savefig(out/f'boundary_{k:03d}.png',dpi=110);fig.savefig(out/f'boundary_{k:03d}.svg');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(14,12))
    for ax,(c,pose) in zip(axes,poses.items()):
        ax.imshow(masks[c][::4,::4],cmap='gray',extent=(0,5760,2880,0),vmin=-.3,vmax=1.4)
        for j,p in enumerate(recipe['people']):
            for k in range(recipe['frame_count']):
                xy=project_erp(world[k,j],pose,5760,2880)[0]
                if k in [0,15,30]:ax.scatter(*xy.T,s=3,label=f'p{p["person_id"]} t={k/10:.1f}')
                ax.plot(*xy.mean(0),'.',color=['blue','purple'][j])
        ax.set_xlim(0,5760);ax.set_ylim(2880,0);ax.set_title(c+' | complete logical ERP, mask only; dots are projected trajectories');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'full_domain_trajectories.svg');fig.savefig(out/'full_domain_trajectories.png',dpi=110);plt.close(fig)
    with (out/'edge_table.csv').open('w') as f:
        fields=['frame','seconds','simultaneous','common_domain','projected_common_crop','cam2_inside','cam2_outside','cam4_inside','cam4_outside'];w=csv.writer(f);w.writerow(fields)
        for r,flag in zip(stats['frames'],stats['dual_edge']['flags']):w.writerow([r['frame_index'],r['frame_index']/10,flag,r['common_physical_points'],r['common_projected_crop_points']]+[r['views'][c][key] for c in poses for key in ['in_view','outside_domain']])
if __name__=='__main__':main()
