"""Reproducible texture comparisons; similarity is evidence, never an identity score."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw


def components(families, edges):
    remaining=set(families); result=[]
    while remaining:
        group={min(remaining)}
        while True:
            grown=group | {v for a,b in edges if a in group or b in group for v in (a,b) if v in remaining}
            if grown==group:break
            group=grown
        result.append(sorted(group));remaining-=group
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();out=a.root/'identity_source_audit';out.mkdir(exist_ok=False)
    proposal=json.loads((a.root/'split.json').read_text());images={};records=[]
    for c in proposal['characters']:
        short=c['character_id'].split('_')[1]
        for ancestor in c['ancestry']:
            m=json.loads(Path(ancestor['manifest']).read_text());key=Path(m['appearance']['body_texture']).name
            records.append(dict(family=short,asset=m['character_id'],texture=key,manifest=ancestor['manifest']))
            if key in images:continue
            im=Image.open(m['appearance']['body_texture']).convert('RGB');w,h=im.size
            rgb=np.array(im.crop((int(.205*w),int(.12*h),int(.405*w),int(.34*h))).resize((256,256)))
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
            images[key]=(rgb,gray-cv2.GaussianBlur(gray,(0,0),5))
    names=sorted(images);sheet=Image.new('RGB',(4*256,280*((len(names)+3)//4)),'white')
    for i,n in enumerate(names):
        sheet.paste(Image.fromarray(images[n][0]),(i%4*256,i//4*280))
        ImageDraw.Draw(sheet).text((i%4*256,i//4*280+258),n,fill='black')
    sheet.save(out/'all_ancestral_faces.jpg')
    comparisons=[]
    for i,x in enumerate(names):
        for y in names[i+1:]:
            if x.split('_')[1]!=y.split('_')[1]:continue
            matrix=np.eye(2,3,dtype=np.float32)
            try:
                score,matrix=cv2.findTransformECC(images[x][1],images[y][1],matrix,cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,150,1e-6),None,3)
            except cv2.error:score=None
            comparisons.append(dict(a=x,b=y,ecc_highpass=score,affine=matrix.tolist()))
    # Human review of original UV detail + ancestry, not a threshold on ECC.
    groups={'female_detail_A':['F01','F03','F04','F05'],
            'female_detail_B':['F02','F04'],
            'male_detail_A':['M01','M04','M05'],
            'male_detail_B':['M02','M03']}
    edges=sorted({tuple(sorted((x,y))) for members in groups.values() for x in members for y in members if x!=y})
    families=[c['character_id'].split('_')[1] for c in proposal['characters']]
    result=dict(status='preliminary_split_rejected_before_generation',records=records,
        method='identical UV face region, grayscale high-pass sigma=5, affine ECC; diagnostic only, not proof of donor identity',
        comparisons=comparisons,conservative_possible_source_groups=groups,
        review_evidence=['Current F04 indian_06 shares possible nose/cheek/lip texture detail with white_01/asian_11; its older indian_03/african_04 versions connect F02.',
          'Male indian_03 has closely aligned facial detail with white_01/white_03 despite darker tone and moustache.',
          'Asian/white female variants have shared fine facial structure; group F01/F03/F05.',
          'M02/M03 retained together conservatively; hair/tone do not establish source independence.'],
        all_ten_components=components(families,edges),
        excluding_F04_components=components([f for f in families if f!='F04'],edges),
        conflict='All five female families become one assignment component through F04 ancestry; cannot put females on both sides with all ten.',
        proposed_resolution=dict(status='awaiting_user_scope_decision',excluded_family='F04',
            train=['F01','F03','F05','M01','M04','M05'],val=['F02','M02','M03']),
        missing_evidence='Official texture pack has no original photographed-person map. Comparisons cannot certify absolute original face-subject independence.',
        preliminary_split=str(a.root/'split.json'),new_layouts_generated=0)
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['all_ten_components'],result['excluding_F04_components'])

if __name__=='__main__':main()
