"""Save mapping design, joint hierarchy, and native kinematic review plots."""


def main():
    import csv
    import json
    from pathlib import Path
    import numpy as np
    from scipy.signal import find_peaks
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from audit_cyclist_bvh import read_bvh

    ROOT=Path(__file__).resolve().parents[2]
    OUT=ROOT/'outputs/audits/dedicated_cycling_mocap_audit'
    mapping=[('Hips','pelvis',0),('LeftUpLeg','left_hip',1),('RightUpLeg','right_hip',2),
     ('Spine','spine1',3),('LeftLeg','left_knee',4),('RightLeg','right_knee',5),('Spine1','spine2',6),
     ('LeftFoot','left_ankle',7),('RightFoot','right_ankle',8),('Spine2','spine3',9),
     ('LeftToeBase','left_foot',10),('RightToeBase','right_foot',11),('Neck','neck',12),
     ('LeftShoulder','left_collar',13),('RightShoulder','right_collar',14),('Head','head',15),
     ('LeftArm','left_shoulder',16),('RightArm','right_shoulder',17),('LeftForeArm','left_elbow',18),
     ('RightForeArm','right_elbow',19),('LeftHand','left_wrist',20),('RightHand','right_wrist',21)]
    with (OUT/'BVH_TO_SMPLX_JOINT_MAPPING.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['bvh_joint','smplx_joint','smplx_index','mapping_status','note'])
        for src,dst,idx in mapping:w.writerow([src,dst,idx,'DESIGN_ONLY','Fit rest frames/bone lengths; do not copy Euler channels or nonroot translations'])

    for participant in [1,2]:
        folder=OUT/f'participant_{participant}'
        d=json.loads((folder/'bvh_audit.json').read_text());selection=json.loads((folder/'clip_selection.json').read_text())
        joints,values,dt=read_bvh(Path(d['path']))
        q=np.load(folder/'joint_positions_native.npy');idx={j['name']:i for i,j in enumerate(joints)}
        assert all(src in idx for src,_,_ in mapping)
        s=selection['selected'];start,end=s['start_frame'],s['end_frame_exclusive'];sl=slice(start,end)
        root=q[sl,0];rel=q[sl]-root[:,None];up=np.array([0.,1.,0.])
        left_axis=(rel[:,idx['LeftUpLeg']]-rel[:,idx['RightUpLeg']]).mean(0)
        left_axis[1]=0;left_axis/=np.linalg.norm(left_axis);forward=np.cross(left_axis,up)
        projected=np.einsum('nji,i->nj',rel,forward)
        t=np.arange(end-start)*dt
        ly=rel[:,idx['LeftFoot'],1];ry=rel[:,idx['RightFoot'],1]
        peaks,_=find_peaks(ly,distance=round(d['fps']*.4),prominence=10)
        cadence=60/(np.median(np.diff(peaks))*dt) if len(peaks)>1 else None
        metrics={'motion_rows_valid':True,'scale_inference':'centimetres: thighs/shins 42-45 units; standing head height about 180 units',
                 'scale_to_m_if_confirmed':.01,'units_explicitly_declared':False,
                 'up_axis_inferred':'+Y','anatomical_left_axis_native':left_axis.tolist(),
                 'anatomical_forward_axis_native':forward.tolist(),'handedness':'Right-handed rotation convention used; no explicit exporter metadata',
                 'left_hand_forward_mean_native':float(projected[:,idx['LeftHand']].mean()),
                 'right_hand_forward_mean_native':float(projected[:,idx['RightHand']].mean()),
                 'both_hands_ahead_all_frames':bool((projected[:,[idx['LeftHand'],idx['RightHand']]]>0).all()),
                 'pelvis_y_range_native':np.ptp(root,axis=0).tolist(),'pedal_peaks':len(peaks),'cadence_peak_interval_rpm':cadence,
                 'major_bone_length_cv_max':max(float(np.std(np.linalg.norm(q[sl,idx[src]]-q[sl,joints[idx[src]]['parent']],axis=1))/np.mean(np.linalg.norm(q[sl,idx[src]]-q[sl,joints[idx[src]]['parent']],axis=1))) for src,_,_ in mapping if src!='Hips'),
                 'posture_estimate':'COMFORTABLE_LIKE' if participant==1 else 'STRAIGHT_ARMS_LIKE',
                 'posture_is_official_segment_label':False,'saddle_distance':'UNAVAILABLE_NO_SADDLE_REFERENCE'}
        (folder/'kinematic_review.json').write_text(json.dumps(metrics,indent=2)+'\n')
        with (folder/'hierarchy.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['index','joint','parent','offset','channels','euler_order','end_site'])
            for i,j in enumerate(joints):w.writerow([i,j['name'],joints[j['parent']]['name'] if j['parent']>=0 else '',j['offset'],' '.join(j['channels']),''.join(c[0] for c in j['channels'] if c.endswith('rotation')),j['end_site']])
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        ax=axes[0,0];ax.plot(t,ly,label='Left ankle');ax.plot(t,ry,label='Right ankle');ax.set(xlabel='Seconds',ylabel='Height relative to pelvis (native units)',title='Alternating pedaling');ax.legend()
        ax=axes[0,1]
        for name in ['LeftFoot','RightFoot']:ax.plot(projected[:,idx[name]],rel[:,idx[name],1],label=name)
        ax.set(xlabel='Anatomical forward',ylabel='Up',title='Ankle sagittal trajectories');ax.axis('equal');ax.legend()
        ax=axes[1,0]
        for name in ['LeftHand','RightHand']:
            pos=rel[:,idx[name]];ax.plot(t,np.linalg.norm(pos-pos.mean(0),axis=1),label=name)
        ax.plot(t,np.linalg.norm(root-root.mean(0),axis=1),label='Pelvis');ax.set(xlabel='Seconds',ylabel='Deviation (native units)',title='Hand/pelvis stability');ax.legend()
        ax=axes[1,1]
        for frame in [0,len(t)//4,len(t)//2]:
            for name,_,_ in mapping:
                j=idx[name];parent=joints[j]['parent']
                if parent<0:continue
                ax.plot(projected[frame,[parent,j]],rel[frame,[parent,j],1],alpha=.5)
        ax.scatter(projected[0,[idx['LeftHand'],idx['RightHand']]],rel[0,[idx['LeftHand'],idx['RightHand']],1],label='Wrists')
        ax.set(xlabel='Anatomical forward',ylabel='Up',title='Native skeleton side view (no SMPL-X fitting)');ax.axis('equal');ax.legend()
        fig.suptitle(f'Participant {participant}: frames [{start}, {end}), approx. {cadence:.1f} RPM; units inferred cm')
        fig.tight_layout();fig.savefig(folder/'kinematic_diagnostic.png',dpi=130);plt.close(fig)
        print(participant,metrics)


if __name__ == "__main__":
    main()
