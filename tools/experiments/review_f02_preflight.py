"""CPU-only physical-domain illustration of the first frozen F02 candidate.

This is a projection diagnostic, not a selected episode or rendered scene.
"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from PIL import Image
from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.camera.erp import project_erp
from densekp_retrain.data.dynamic_scene import scene_trajectory,geometry_paths
from densekp_retrain.data.boxes import crop_from_box
from densekp_retrain.imaging.roi_planning import plan_native_crop_region
from densekp_retrain.data.mixed_road_pass import edge_summary

def main():
    root=ROOT/'outputs/experiments/f02_mtb_mixed_scene_a1_20261010'
    plan=json.loads((root/'plan_grouped/plan.json').read_text())
    ep=plan['episodes'][0];cfg=plan['config'];q=plan['candidate_sets'][0]['candidates'][0]['parameters']
    asset=json.loads(Path(ep['people'][0]['asset_manifest']).read_text())
    ix=ep['people'][0]['asset_frames']
    paths=geometry_paths(asset)
    dense=np.load(paths['densekp'],mmap_mode='r')
    body=np.load(paths['body_vertices'],mmap_mode='r')
    cloth=np.load(paths['cloth_vertices'],mmap_mode='r')
    mats=scene_trajectory(ep['episode_timeline'],[q['target_x_m'],q['target_y_m'],cfg['placement_height_m']],[0,1],q['cycling_speed_m_s'],np.array(cfg['fixed_scene_rotation']))
    poses,_,_=rig_poses(ep['rig']['origin_m'],ep['rig']['yaw_deg'],ep['rig']['baseline_m'])
    masks={c:np.asarray(Image.open(p))>0 for c,p in cfg['physical_masks'].items()}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    out=root/'projection_review';out.mkdir(exist_ok=True)
    rows=[]
    for k in range(ep['frame_count']):
        M=mats[k];world=dense[ix[k]]@M[:3,:3].T+M[:3,3]
        inside={c:point_in_view(world,pose,c,masks[c])[-1] for c,pose in poses.items()}
        rows.append(dict(frame_index=k,common_physical_points=int(np.logical_and(*inside.values()).sum()),
            views={c:dict(in_view=int(v.sum()),outside_domain=int((~v).sum())) for c,v in inside.items()}))
    (out/'target_only_edge.json').write_text(json.dumps(dict(status='diagnostic_target_only_not_selected_episode',
        candidate_seed=plan['candidate_sets'][0]['candidates'][0]['child_seed'],parameters=q,
        summary=edge_summary(rows,ep['episode_timeline']),frames=rows),indent=2,allow_nan=False)+'\n')
    for k in [0,9,15,30]:
        fig,axes=plt.subplots(1,2,figsize=(15,6))
        M=mats[k];to_world=lambda vertices:vertices[ix[k]]@M[:3,:3].T+M[:3,3]
        for ax,(c,pose) in zip(axes,poses.items()):
            xy,_,_,_,inside=point_in_view(to_world(dense),pose,c,masks[c])
            envelope=np.concatenate([project_erp(to_world(v),pose,5760,2880)[0] for v in (body,cloth)])
            hull=cv2.convexHull(envelope.astype(np.float32))[:,0]
            ax.imshow(masks[c][::4,::4],cmap='gray',vmin=0,vmax=1,extent=(0,5760,2880,0),interpolation='nearest')
            ax.fill(hull[:,0],hull[:,1],color='#12a7d5',alpha=.28,label='body+cloth projected convex hull (proxy)')
            ax.plot(np.r_[hull[:,0],hull[0,0]],np.r_[hull[:,1],hull[0,1]],color='#12a7d5',lw=1)
            ax.scatter(*xy[inside].T,s=7,color='#60d75f',label='R138 inside physical mask')
            ax.scatter(*xy[~inside].T,s=8,color='#f4604d',marker='x',label='R138 outside physical mask')
            keep=[]
            for part in (body,cloth):
                z,_,_,_,b=point_in_view(to_world(part),pose,c,masks[c]);keep.append(z[b])
            keep=np.concatenate(keep)
            if len(keep):
                box=np.r_[keep.min(0),keep.max(0)];center,scale=crop_from_box(box,image_size_wh=(5760,2880),crop_policy='deployment',boundary_policy='erp_reject_seam',min_side_px=2)
                side=200*scale;ax.add_patch(Rectangle(center-side/2,side,side,fill=False,color='#d2ff52',label='target projected native square'))
            try:
                r=plan_native_crop_region({'body':project_erp(to_world(body),pose,5760,2880)[0],
                                           'cloth':project_erp(to_world(cloth),pose,5760,2880)[0]},(5760,2880))['roi_xyxy']
                ax.add_patch(Rectangle((r[0],r[1]),r[2]-r[0],r[3]-r[1],fill=False,color='#ffbe4c',linestyle='--',label='target-only planned ROI'))
            except ValueError:
                pass
            contour=cv2.findContours(masks[c].astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
            for v in contour:
                p=v[:,0];ax.plot(p[:,0],p[:,1],color='yellow',lw=.6)
            ax.set_xlim(0,5760);ax.set_ylim(2880,0);ax.set_title(f'{c}, frame {k}, inside/outside={int(inside.sum())}/{int((~inside).sum())}')
            ax.set_xlabel('complete ERP u (px)');ax.set_ylabel('complete ERP v (px)')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
        fig.suptitle('F02 candidate 0, target only; no RGB/render; walker ROI later exceeds raster')
        fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(out/f'target_only_{k:03d}.png',dpi=110);plt.close(fig)

if __name__=='__main__':main()
