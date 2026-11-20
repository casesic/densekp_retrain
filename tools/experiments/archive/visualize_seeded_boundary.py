"""Small CPU diagrams from an already preflighted single-person episode."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from densekp_retrain.camera.domain import point_in_view


def visualize(episode,out):
    episode=Path(episode);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    recipe=json.loads((episode/'recipe.json').read_text())
    if len(recipe['people'])!=1:raise ValueError('single_person_diagram_only')
    matrices=np.load(episode/'scene_matrices.npy')[:,0]
    xyz=np.load(episode/'world_densekp.npy')[:,0]
    rig=json.loads((episode/'rig.json').read_text())
    masks={c['id']:np.asarray(Image.open(episode/'calibration'/f"{c['id']}_valid_mask.png"))>0 for c in rig['cameras']}
    planned=json.loads((episode/'planned_regions.json').read_text())
    fig,ax=plt.subplots(figsize=(7,6));road=recipe['road_bounds_xyxy'];ax.axvspan(road[0],road[2],color='#eeeeee',label='planned road corridor')
    path=matrices[:,:2,3];ax.plot(path[:,0],path[:,1],'-o',markersize=2,color='#c44',label='scene-owned rider path')
    for k in (0,15,30):ax.annotate(f't={k/10:.1f}s',path[k],xytext=(5,5),textcoords='offset points')
    for camera in rig['cameras']:
        M=np.array(camera['camera_to_world']);x,y=M[:2,3]
        ax.scatter([x],[y],marker='^',s=110,label=camera['id'])
        ax.arrow(x,y,M[0,2]*.3,M[1,2]*.3,head_width=.05,length_includes_head=True)
    ax.set(xlabel='scene X (m)',ylabel='scene Y (m)',title='Rider trajectory and fixed dual-camera rig')
    ax.text(.02,.02,'Physical overlap is determined in each calibrated ERP mask; no surveyed floor seam is asserted.',transform=ax.transAxes,fontsize=7)
    ax.axis('equal');ax.grid(alpha=.2);ax.legend(loc='best');fig.tight_layout();fig.savefig(out/'topdown.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for row,k in enumerate((0,15,30)):
        for col,camera in enumerate(rig['cameras']):
            c=camera['id'];pose=np.array(camera['camera_to_world']);mask=masks[c]
            xy,_,_,_,inside=point_in_view(xyz[k],pose,c,mask)
            thumb=cv2.resize(mask.astype(np.uint8),(720,360),interpolation=cv2.INTER_NEAREST)
            ax=axes[row,col];ax.imshow(thumb,cmap='gray',vmin=0,vmax=1,extent=(0,5760,2880,0))
            ax.scatter(xy[~inside,0],xy[~inside,1],s=4,c='#ff5b50',label='outside physical domain')
            ax.scatter(xy[inside,0],xy[inside,1],s=4,c='#36cf78',label='inside physical domain')
            x0,y0,x1,y1=planned[k][c]['roi_xyxy']
            ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#39aaff',linewidth=1))
            ax.set(xlim=(0,5760),ylim=(2880,0),title=f'{c} t={k/10:.1f}s: in {inside.sum()}, out {(~inside).sum()}')
    axes[0,0].legend(fontsize=7,loc='lower left');fig.suptitle('Full ERP coordinates; blue = planned high-quality ROI, green/red = R138 physical support')
    fig.savefig(out/'physical_projection.png',dpi=120);plt.close(fig)
    (out/'diagram_manifest.json').write_text(json.dumps(dict(episode_ref=str(episode.resolve()),frames=[0,15,30],
        projection='actual calibrated normal ERP and physical valid mask',topdown_seam='not asserted',
        image_background='calibration mask only; not rendered RGB'),indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episode',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();visualize(a.episode,a.output)


if __name__=='__main__':main()
