"""Rank 10-second native-BVH windows by measurable cycling stability."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.signal import detrend, find_peaks

def main():
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path);a=p.parse_args()
    d=json.loads((a.audit/'bvh_audit.json').read_text());q=np.load(a.audit/'joint_positions_native.npy')
    ids={j['name']:i for i,j in enumerate(d['joints'])};fps=d['fps']
    n=lambda name:q[:,ids[name]]
    pelvis=n('Hips');left=n('LeftFoot')-pelvis;right=n('RightFoot')-pelvis
    def angle(x,y,z):
        u=x-y;v=z-y
        return np.degrees(np.arccos(np.clip((u*v).sum(1)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)),-1,1)))
    elbows=np.c_[angle(n('LeftArm'),n('LeftForeArm'),n('LeftHand')),
                  angle(n('RightArm'),n('RightForeArm'),n('RightHand'))]
    rows=[];window=round(10*fps);stride=round(5*fps)
    for start in range(round(10*fps),len(q)-window,stride):
        end=start+window;sl=slice(start,end)
        l=detrend(left[sl,1]);r=detrend(right[sl,1]);signal=l-r
        freq=np.fft.rfftfreq(window,1/fps);power=abs(np.fft.rfft(signal*np.hanning(window)))**2
        band=(freq>=.8)&(freq<=2.1);peak=np.where(band)[0][np.argmax(power[band])]
        cadence=freq[peak]*60;peakratio=power[max(1,peak-1):peak+2].sum()/max(power[1:].sum(),1e-12)
        corr=float(np.corrcoef(l,r)[0,1]) if min(l.std(),r.std())>1e-6 else 0
        handstd=max(np.linalg.norm((n(side+'Hand')-pelvis)[sl].std(0)) for side in ['Left','Right'])
        rootstd=np.linalg.norm(pelvis[sl].std(0))
        step=max(np.linalg.norm(np.diff(n(side+'Foot')[sl],axis=0),axis=1).max() for side in ['Left','Right'])
        amplitude=min(np.ptp(l),np.ptp(r))
        elbow=elbows[sl].mean(0)
        torso=(n('Neck')-pelvis)[sl].mean(0)
        lean=np.degrees(np.arctan2(np.linalg.norm(torso[[0,2]]),torso[1]))
        # Units deliberately retained as native; anatomical scale is audited separately.
        eligible=bool(corr<-.65 and peakratio>.65 and amplitude>8 and handstd<5 and rootstd<3 and step<15)
        rows.append(dict(start_frame=start,end_frame_exclusive=end,start_sec=start/fps,end_sec=end/fps,
                         cadence_rpm=cadence,cycle_spectral_concentration=peakratio,left_right_vertical_correlation=corr,
                         foot_vertical_amplitude_native=amplitude,hand_relative_std_native=handstd,
                         pelvis_std_native=rootstd,max_foot_step_native=step,
                         left_elbow_deg=elbow[0],right_elbow_deg=elbow[1],torso_lean_deg=lean,
                         eligible=eligible,score=handstd+rootstd+abs(cadence-75)/30+(1-peakratio)*5))
    ordered=sorted(rows,key=lambda r:(not r['eligible'],r['score']))
    with (a.audit/'clip_windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(ordered)
    result={'selected':ordered[0],'eligible_windows':sum(r['eligible'] for r in rows),
            'selection_verified':ordered[0]['eligible'],'posture_label':'UNANNOTATED_KINEMATIC_ESTIMATE_REQUIRED',
            'saddle_contact':'NOT_DIRECTLY_VERIFIABLE_NO_BICYCLE_OR_SADDLE_MARKER',
            'body_forward':'Must infer from pelvis/hip axes, not fixed camera direction',
            'retarget_performed':False}
    (a.audit/'clip_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
