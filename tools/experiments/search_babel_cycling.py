"""Audit all BABEL v1 annotation sets, preserving ambiguity and provenance."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('/home/yy/datasets/BABEL/babel_v1.0_release')
OUT = ROOT/'outputs/audits/cycling_motion_audit'
SPLITS = ['train','val','test','extra_train','extra_val']
BROAD = re.compile(r'bik|bicycl|cycl|pedal|peddl|\bride\b|\briding\b|motor|horse|scooter',re.I)
HIGH = re.compile(r'\b(?:rid(?:e|ing) (?:a )?bike|bicycle|bicycling|cycling|bike riding)\b',re.I)
EXCLUDE = re.compile(r'\b(?:motorcycle|motorbike|horse|horseback|scooter)\b',re.I)

def annotations(record):
    for key in ('seq_ann','frame_ann','seq_anns','frame_anns'):
        value = record.get(key)
        if value is None: continue
        groups = [value] if isinstance(value,dict) else value
        if not isinstance(groups,list): raise TypeError((key,type(value)))
        for index, group in enumerate(groups):
            for label in group['labels']:
                yield key,index,group,label

def label_text(label):
    return ' '.join([label.get('raw_label') or '',label.get('proc_label') or ''] + (label.get('act_cat') or []))

def write_csv(name, rows, fields):
    with (OUT/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)

def main():
    OUT.mkdir(exist_ok=True)
    data, contexts, schema = {},defaultdict(list),{}
    unique, paths = set(),set()
    for split in SPLITS:
        file=SOURCE/f'{split}.json'; data[split]=json.loads(file.read_text())
        types=defaultdict(Counter); label_keys=Counter(); nulls=Counter(); count=0
        for sid,rec in data[split].items():
            assert str(rec['babel_sid'])==sid
            unique.add(sid);paths.add(rec['feat_p'])
            for key,val in rec.items(): types[key][type(val).__name__]+=1
            for key,index,group,label in annotations(rec):
                count+=1;label_keys.update(label.keys())
                nulls.update(k for k,v in label.items() if v is None)
                contexts[sid].append({'split':split,'level':key,'text':label_text(label),
                                      'raw_label':label.get('raw_label')})
        schema[split]={'records':len(data[split]),'labels':count,'field_types':dict(types),
                       'label_key_counts':dict(label_keys),'null_label_fields':dict(nulls),
                       'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    rows=[]
    for split,records in data.items():
        for sid,rec in records.items():
            for key,index,group,label in annotations(rec):
                text=label_text(label)
                if not BROAD.search(text):continue
                confidence='HIGH' if HIGH.search(text) else ('MEDIUM' if re.search(r'bik|pedal|peddl',text,re.I) else 'LOW')
                exclusion=''
                if re.search(r'back\s*pedal|back\s*peddl',text,re.I):exclusion='BACKWARD_MOVEMENT_NOT_CYCLING'
                elif re.search(r'bicycle arms',text,re.I):exclusion='ARM_CIRCLES_NOT_BICYCLE_RIDING'
                elif EXCLUDE.search(text) and not HIGH.search(text):exclusion='NON_BICYCLE_LABEL'
                elif re.search(r'step ride side',text,re.I):exclusion='SIDEWAYS_STEP_CONTEXT'
                elif EXCLUDE.search(rec['feat_p'].replace('_',' ')):exclusion='SOURCE_MOTORCYCLE_CONFLICT_REQUIRES_REVIEW' if confidence=='HIGH' else 'NON_BICYCLE_SOURCE_CONTEXT'
                if not exclusion and confidence=='LOW':exclusion='GENERIC_RIDE_UNRESOLVED'
                start=label.get('start_t');end=label.get('end_t')
                timing='EXPLICIT_FRAME_INTERVAL'
                if key.startswith('seq'):
                    start=0 if start is None else start;end=rec['dur'] if end is None else end
                    timing='SEQUENCE_EXTENT_NOT_VERIFIED_CONTINUOUS_CYCLING'
                duration=None if start is None or end is None else end-start
                assert duration is None or duration>=0
                row=dict(split=split,babel_sid=sid,feat_p=rec['feat_p'],duration_sec=rec['dur'],
                         annotation_level=key,raw_label=label.get('raw_label'),proc_label=label.get('proc_label'),
                         act_cat=json.dumps(label.get('act_cat')),start_t=start,end_t=end,
                         cycling_confidence=confidence,exclusion_reason=exclusion,
                         source_dataset=rec['feat_p'].split('/')[0],source_collection=rec['feat_p'].split('/')[1],
                         segment_duration=duration,timing_basis=timing,seg_id=label.get('seg_id'),
                         annotation_group_index=index,babel_lid=group.get('babel_lid'),
                         context=json.dumps(contexts[sid]),motion_inspection='NEED_MOTION_INSPECTION',
                         acquisition_cost='UNKNOWN_NO_PACKAGE_SIZE_VERIFIED')
                rows.append(row)
    fields=['split','babel_sid','feat_p','duration_sec','annotation_level','raw_label','proc_label','act_cat',
            'start_t','end_t','cycling_confidence','exclusion_reason','source_dataset','source_collection',
            'segment_duration','timing_basis','seg_id','annotation_group_index','babel_lid','context',
            'motion_inspection','acquisition_cost']
    write_csv('babel_cycling_candidates.csv',rows,fields)
    # Lexical HIGH candidates are kept as review priorities even when quarantined.
    high=[r for r in rows if r['cycling_confidence']=='HIGH' and r['exclusion_reason']!='ARM_CIRCLES_NOT_BICYCLE_RIDING']
    high.sort(key=lambda r:(bool(r['exclusion_reason']),
                           any(EXCLUDE.search(c['text']) for c in contexts[r['babel_sid']]),
                           -(r['segment_duration'] or 0),-r['duration_sec'],int(r['babel_sid'])))
    top=[];seen=set()
    for row in high:
        if row['feat_p'] in seen:continue
        seen.add(row['feat_p']);r=row.copy();r['rank']=len(top)+1
        r['reason']='ANNOTATION_ONLY; '+(row['exclusion_reason'] or 'explicit bicycle label')+'; NEED_MOTION_INSPECTION for legs/hands/pelvis; download cost UNKNOWN'
        top.append(r)
        if len(top)==10:break
    write_csv('top_cycling_candidates.csv',top,['rank','babel_sid','feat_p','source_dataset','raw_label',
              'start_t','end_t','segment_duration','reason','split','annotation_level','timing_basis','exclusion_reason'])
    # Compare normalized collection/subject/sequence stems, never numeric IDs alone.
    def normalized(path):
        parts=path.replace('\\','/').split('/')
        tail='/'.join(parts[-3:]);return re.sub(r'_(poses|stageii)\.npz$','',tail).lower()
    previous=json.loads((OUT/'local_motion_audit.json').read_text())
    overlaps=[]
    for r in top:
        for m in previous['motions']:
            if normalized(r['feat_p'])==normalized(m['motion_info'][2]):
                overlaps.append({'babel_sid':r['babel_sid'],'bedlam_motion':m['motion_path']})
    inventory=(OUT/'local_inventory.txt').read_text().splitlines()
    local_matches=[p for p in inventory if p.endswith('.npz') and any(normalized(p)==normalized(r['feat_p']) for r in top)]
    summary={'TOTAL_BABEL_SEQUENCES':len(unique),'total_split_records':sum(len(x) for x in data.values()),
             'unique_feat_paths':len(paths),'CYCLING_RAW_MATCH_COUNT':len(rows),
             'HIGH_CONFIDENCE_CYCLING_COUNT':len(high),
             'MEDIUM_CONFIDENCE_CYCLING_COUNT':sum(r['cycling_confidence']=='MEDIUM' and not r['exclusion_reason'] for r in rows),
             'accepted_high_count':sum(not r['exclusion_reason'] for r in high),
             'raw_confidence_counts_before_context_filter':dict(Counter(r['cycling_confidence'] for r in rows)),
             'excluded_rows':sum(bool(r['exclusion_reason']) for r in rows),
             'source_raw_match_counts':dict(Counter(r['source_dataset'] for r in rows)),
             'source_high_label_counts':dict(Counter(r['source_dataset'] for r in high)),
             'top':top,'bedlam_local_overlaps':overlaps,'BEDLAM_OVERLAP_FOUND':'YES' if overlaps else 'UNKNOWN',
             'AMASS_MOTION_LOCAL':bool(local_matches),'local_matches':local_matches,
             'selection_status':'REVIEW_ONLY_NO_CYCLING_DOWNLOAD_RECOMMENDED',
             'schema':schema}
    (OUT/'babel_cycling_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('top','schema')},indent=2))

if __name__=='__main__':main()
