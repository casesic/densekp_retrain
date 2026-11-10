"""Read-only buffer/point audit plus cost summary for the completed trial."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
from densekp_retrain.imaging.source import read_ds_source
from densekp_retrain.camera.erp import project_erp
from densekp_retrain.imaging.regions import require_crop_in_region,require_point_neighborhoods
OUT=ROOT/'outputs/experiments/direct_erp_roi_trial_20261009'

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def inventory(root):return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p)}

def main():
    began=time.monotonic();f=read(OUT/'frozen_comparison.json');world=np.load(Path(f['old_episode'])/'world_densekp.npy')[0].reshape(-1,3);details=[];coverage=[]
    for c,info in f['cameras'].items():
        full=read_ds_source(OUT/'buffers/full/_temporary_ds_source/source',c);roi=read_ds_source(OUT/'buffers/roi/_temporary_ds_source/source',c)
        x0,y0,x1,y1=info['roi_xyxy'];xy,distances=project_erp(world,np.array(info['camera_to_world']),5760,2880)
        require_point_neighborhoods(xy,info['roi_xyxy'],(5760,2880))
        labels={r:dict(np.load(OUT/r/c/'annotations.npz')) for r in ['A','B','C']}
        inside=np.zeros(full.instance.shape,bool);inside[y0:y1,x0:x1]=True
        outside_actual_person=int(((np.rint(full.instance)==1)&~inside).sum())
        d=roi.ray_range_m-full.ray_range_m[y0:y1,x0:x1];finite=np.isfinite(d)
        body_neighborhood=[]
        for i,(x,y) in enumerate(xy):
            ix,iy=int(np.floor(x)),int(np.floor(y));row={}
            for name,depth,index,dx,dy in [('B',full.ray_range_m,full.instance,0,0),('C',roi.ray_range_m,roi.instance,x0,y0)]:
                sd=depth[iy-dy-2:iy-dy+3,ix-dx-2:ix-dx+3];si=index[iy-dy-2:iy-dy+3,ix-dx-2:ix-dx+3];matched=(sd>0)&(np.rint(si)==1)
                row[name]=dict(matching_ID_pixels=int(matched.sum()),minimum_abs_range_error_m=float(np.abs(sd[matched]-distances[i]).min()) if matched.any() else None,
                    visible=bool(labels[name]['visible'][0,i]),surface_visible=bool(labels[name]['surface_visible'][0,i]),in_view=bool(labels[name]['in_view'][0,i]))
            source_diff=row['B']['surface_visible']!=row['C']['surface_visible'];visible_diff=row['B']['visible']!=row['C']['visible']
            if source_diff or visible_diff:
                details.append(dict(camera=c,semantic_id=i,xy=xy[i].tolist(),range_m=float(distances[i]),B=row['B'],C=row['C'],
                    reason='ID support changed' if row['B']['matching_ID_pixels']!=row['C']['matching_ID_pixels'] else 'range test crossed strict4cm threshold'))
            body_neighborhood.append(row)
        rejected=False
        try:require_crop_in_region([100,100],.5,info['roi_xyxy'],(5760,2880))
        except ValueError as e:rejected=str(e)=='crop_outside_high_quality_support'
        assert rejected and outside_actual_person==0
        assert np.array_equal(labels['B']['xy'],labels['C']['xy']) and np.array_equal(labels['B']['in_view'],labels['C']['in_view'])
        for b in info['diagnostic_crops']:require_crop_in_region(b['center'],b['scale'],info['roi_xyxy'],(5760,2880))
        coverage.append(dict(camera=c,raw_full_person_pixels_outside_frozen_roi=outside_actual_person,all_138_neighborhoods_inside=True,
            raw_ROI_ID_mismatch_pixels=int((np.rint(roi.instance)!=np.rint(full.instance[y0:y1,x0:x1])).sum()),
            raw_ROI_depth_max_abs_m=float(np.abs(d[finite]).max()),raw_ROI_depth_exact_fraction=float((d[finite]==0).mean()),
            outside_crop_rejected=rejected,geometry_xy_exact=True,in_view_exact=True,neighborhoods=body_neighborhood))
    write(OUT/'visibility_difference_details.json',dict(changed_points=details,changed_count=len(details),rule='5x5 same-person raw range; strict abs difference<.04m; then calibrated in_view'))
    write(OUT/'ROI_support_validation.json',dict(status='passed',checks=coverage,validation_uses_full_buffer_as_comparator=True,
        C_label_production_uses_full_buffer=False,elapsed_seconds=time.monotonic()-began))
    # Explicit shared-render attribution; production costs exclude comparison-only reads.
    ab=read(OUT/'AB_processing.json')['times'];c=read(OUT/'C/manifest.json')['timings'];full_run=read(OUT/'full_process.json');region_run=read(OUT/'region_process.json');tiny_run=read(OUT/'tiny_process.json')
    At=sum(sum(v for k,v in row.items() if k.startswith('A_') or k=='common_exr_and_display_rgb_read') for row in ab)
    Bt=sum(sum(v for k,v in row.items() if k.startswith('B_') or k=='common_exr_and_display_rgb_read') for row in ab)
    Ct=sum(sum(v for k,v in row.items() if k!='camera') for row in c)
    rt=read(OUT/'loader_readback.json')['route_times'];pack={r['route']:r['metadata_masks_pack_and_Dataset_init_seconds']+r['ten_Dataset_reads_affine_checks_and_crop_exports_seconds'] for r in rt}
    render_times={p.stem.split('_')[0]:read(p)['elapsed_seconds'] for p in OUT.glob('*_render_summary.json')}
    a_cost=full_run['wall_seconds']+At;b_cost=full_run['wall_seconds']+Bt;c_cost=region_run['wall_seconds']+Ct
    disk={r:sum(p.stat().st_size for p in (OUT/r).rglob('*') if p.is_file()) for r in ['A','B','C']}
    images={r:sum((OUT/r/name/'normal_erp.png').stat().st_size for name in ['cam2','cam4']) for r in disk}
    events=[json.loads(s) for s in (OUT/'render_calls.jsonl').read_text().splitlines()];completed=[e for e in events if e['event']=='completed'];assert len(completed)==7 and len([e for e in events if e['event']=='started'])==7
    total=full_run['wall_seconds']+region_run['wall_seconds']+tiny_run['wall_seconds']+read(OUT/'scene_settings.json')['elapsed_seconds'];assert total<1200
    write(OUT/'costs.json',dict(scope='one fixed scene time, two cameras; no prediction or training',actual_render_calls=7,actual_render_process_wall_seconds_including_inspect_and_tiny=total,
        render_stage_budget_seconds=1200,remaining_budget_seconds=1200-total,
        full_render_shared_but_charged_to_A_and_B_each=full_run['wall_seconds'],C_independent_render_process_seconds=region_run['wall_seconds'],tiny_validation_only_seconds=tiny_run['wall_seconds'],
        postprocess_production_seconds=dict(A=At,B=Bt,C=Ct),cold_production_seconds=dict(A=a_cost,B=b_cost,C=c_cost),
        cold_speedup_vs_A=dict(B=a_cost/b_cost,C=a_cost/c_cost),C_speedup_vs_B=b_cost/c_cost,
        cold_production_plus_ten_diagnostic_readbacks_seconds={r:v+pack[r] for r,v in [('A',a_cost),('B',b_cost),('C',c_cost)]},
        pack_and_ten_readbacks_seconds=pack,route_final_bytes_including_diagnostic_crops=disk,normal_ERP_PNG_bytes=images,
        initialization_seconds={r:read(OUT/(stage+'_setup.json'))['seconds'] for r,stage in [('A','full'),('B','full'),('C','region')]},
        render_calls_combined_render_compositor_EXR_PNG=[dict(stage=e['stage'],camera=e['camera'],seconds=e['seconds']) for e in completed],
        GPU_kernel_time='not separately measured; render/compositor/I/O combined',
        GPU_peak_sampled_MiB=dict(full=full_run['gpu_memory_sampled_peak_MiB'],tiny=tiny_run['gpu_memory_sampled_peak_MiB'],region=region_run['gpu_memory_sampled_peak_MiB']),
        cold_scope='Blender process wall includes startup/scene/assets/exit and <=2s monitor polling quantization; postprocess sums measured production substeps, excludes trial-only comparisons',
        warm_scope='no full later scene time measured; per-camera second calls are warm within a stage, not a continuous-motion throughput guarantee',
        cached_background='not measured, not subtracted; same scene/camera and lighting required; moving actor shadows/reflections/background contribution need separate invalidation'))
    print(json.dumps(read(OUT/'costs.json'),indent=2))

if __name__=='__main__':main()
