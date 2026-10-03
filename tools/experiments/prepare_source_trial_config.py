"""Prepare a bounded next-run recipe and audit immutable regression references.

Does not load a model, generate a scene, infer, or train. Missing candidate data
keeps ready_for_training false instead of substituting the old integration rows.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    root=Path(sys.argv[1]).resolve();old=ROOT/'outputs/experiments/mixed_riders_pedestrians_20260930'
    riding=ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930'
    base=old/'original_baseline';checks=[];counts=[]
    for label,path in [('riding_A',riding/'candidate_index.json'),('mixed_B',old/'candidate_index.json')]:
        if path.read_bytes()!=(base/f'{label}_frozen_index.json').read_bytes():raise ValueError('Old baseline index changed')
        rows=json.loads(path.read_text())['rows'];files=[];supervised=occluded=0
        for i,row in enumerate(rows):
            pred=base/f'{label}_{i:03d}'/'prediction.npz';z=np.load(pred)
            if z['prediction_raw'].shape!=(138,3) or not np.isfinite(z['prediction_raw']).all():
                raise ValueError('Old raw output changed contract')
            supervised+=int(z['supervised'].sum());occluded+=int((z['supervised']&~z['visible']).sum())
            st=pred.stat();files.append(dict(path=str(pred),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
        checks.append(dict(label=label,index=str(path),frozen_copy_equal=True,predictions=files))
        counts.append(dict(batch=label,rows=len(rows),supervised_points=supervised,occluded_supervised_points=occluded))
    checkpoint=ROOT/'assets_external/densekp.ckpt';st=checkpoint.stat()
    loaded=json.loads((base/'checkpoint_load.json').read_text())
    if (st.st_size,st.st_mtime_ns)!=(loaded['bytes'],loaded['mtime_ns']):raise ValueError('Original checkpoint file changed')
    write(root/'regression_reference_audit.json',dict(status='read_only_checks_passed',checks=checks,counts=counts,
        checkpoint_stat_matches_previous_strict_load=True,previous_strict_load=loaded,
        checkpoint_reloaded_this_turn=False,predictions_recomputed=False,
        limitation='No content hash; exact index bytes and raw-array contracts checked. No claim to cryptographic file integrity.'))
    training=dict(status='recommendation_pending_generated_data',ready_for_training=False,
        init_checkpoint=str(checkpoint),strict_load=True,resume_checkpoint=None,
        trainable='all backbone and head parameters',optimizer='AdamW',learning_rate=1e-6,weight_decay=1e-4,
        batch_size=1,accumulate_grad_batches=1,precision='bf16-mixed',metric_precision='float32',
        augmentation=False,shuffle=True,seed=930,max_steps=400,validate_every_updates=100,
        early_stopping=False,learning_rate_schedule='constant',gradient_clipping=False,
        supervision='in_view & inside_full_crop',coordinates='full 256-square; backbone reads center 192 columns',
        output_third_value='preserved raw, never confidence/weight/filter',
        validation='all frozen val rows at steps 0/100/200/300/400; FP32 endpoint metrics',
        save='Keep each scheduled checkpoint; final step 400 is the predeclared primary comparison, no best-of-many claim',
        reload='strict load each saved checkpoint; fixed prediction max_abs <=1e-5; finite gradients and actual optimizer updates required',
        sample_exposures=400,unique_train_rows=None,equivalent_passes_formula='400 / actual frozen train row count',
        hardware='one idle RTX 3090 24GB; preflight again before compute',
        prior_measured_peak_bytes=17915637760,prior_memory_evidence=str(ROOT/'outputs/experiments/cycling_train_smoke_20260929/report.json'),
        runtime_estimate='training + scheduled FP32 validation + saving/reloading: provision 15–40 minutes; unmeasured for new data',
        prerequisites=['approved frozen identity/source split','actual scene/loader acceptance','frozen train/val row indices',
                       'new original-weight baseline','source/intersection validator passed'],
        implementation_notes=['DenseKP.training_step manually zeroes/steps each batch; accumulation >1 is unsupported.',
                              'Use existing densekp.core.training_audit.CheckUpdates callback and strict reload probe.',
                              'Build new dataset registry entries only from accepted indices; old 51 rows never substitute for missing data.',
                              'No training command is executable yet: candidate NPZ/index files do not exist.'])
    training['effect_criteria_proposed']=dict(units='full 256-square pixels',
        primary='val rear-target and val occluded-supervised point means both improve >=5% versus frozen original weights',
        non_regression='complete-control and visible-supervised means worsen no more than max(0.5px,5%); inspect action groups too',
        averaging='point mean and equal-person mean both reported; paired occlusion delta diagnostic reported separately',
        workflow='all rows scored, no nonfinite denominator reduction, actual 400 optimizer updates, reload <=1e-5',
        empty_group='null / criterion not established, never pass at zero error',
        limitation='engineering go/no-go suggestion, not statistical significance; only two pedestrian sources, no test set')
    write(root/'training_recommendation.json',training)
    recipe=dict(status='proposal_not_frozen_not_generated',split_proposal=str(root/'split_proposal_v2.json'),
        pilots='one train layout and one val layout before expansion',
        train=dict(target_boxes=[200,400],proposed_layouts=36,people_per_layout=4,expected_boxes=252),
        val=dict(target_boxes=[80,120],proposed_layouts=20,people_per_layout=3,expected_boxes=100),
        images_per_layout=2,total_expected_erp_images=112,
        scenes=['cycling_forest','urban_street_assets'],forest_fraction=.25,
        render=dict(erp_size=[1536,768],samples=24,seed=930,lighting='native',calibration='20260320',synthetic_baseline_m=.18),
        coverage='Rotate gender/identity/action across front/rear/complete/edge slots, vary distance/yaw/placement materially. No near-identical full Cartesian expansion.',
        rear_target_fraction='aim near one-third; actual boxes measured after rendering',
        paired_occlusion='2 pairs per split within layout budget; same target/world transform/camera/given box/affine; change only occluder',
        expected_render_cost='Approximately 2–3 GPU hours using previous street/forest pair timings; reassess after both pilots, do not silently multiply scale.',
        acceptance='existing numerical gates + actual ERP/256-square/256x192 appearance; selection before prediction',
        counts_are='estimates, not generated rows; correlated views/layouts do not increase independent identity/pose sources')
    write(root/'generation_recipe_proposal.json',recipe)
    write(root/'status.json',dict(status='awaiting_identity_scope_resolution',utc=datetime.now(timezone.utc).isoformat(),
        independent_pedestrian_sources_available=2,independent_cycling_sources_available=2,
        new_source_selected_postures=3,new_train_rows=0,new_val_rows=0,new_erp_images=0,new_predictions=0,
        retained_regression_rows=51,training_started=False,
        invalidated_preliminary_split=str(root/'split.json'),active_proposal=str(root/'split_proposal_v2.json'),
        blocking_issue='F04 ancestry bridges possible female face-source groups; proposed whole-family exclusion needs scope resolution'))
    print(counts)

if __name__=='__main__':main()
