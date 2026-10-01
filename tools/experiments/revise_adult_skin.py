"""Preserve similar-face failures; revise skin within the same synthetic identity family."""
import argparse,json,sys,zipfile
from copy import deepcopy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.experiments.bind_adult_grooms import scalp,PILOT
from tools.experiments.prepare_cycling_characters import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('assets',type=Path)
    p.add_argument('--changes',nargs='+',default=['F04:skin_f_african_04_ALB','F05:skin_f_asian_11_ALB'])
    p.add_argument('--revision',default='skin_revision');a=p.parse_args();root=a.assets.resolve()
    report=json.loads((root/'report.json').read_text());write_json(root/f'report_before_{a.revision}.json',report)
    uniform=root/f'{a.revision}_inputs';uniform.mkdir();comparisons=[];history=[]
    for change in a.changes:
        short,skin=change.split(':')
        item=next(c for c in report['characters'] if c['character_id'].startswith(f'cycling_{short}_v'))
        parent=Path(item['manifest']);m=deepcopy(json.loads(parent.read_text()));version=m['version']+1
        cid=f'cycling_{short}_v{version}';out=root/'characters'/cid;out.mkdir()
        source=dict(m['texture_source']);source['source_id']=skin
        source['member']=f'bedlam_body_textures_meshcapade/smpl/MC_texture_skintones/female/skin/{skin}.png'
        source['original_preview_member']=f'bedlam_body_textures_meshcapade/smpl/MC_texture_skintones/female/skin/previews/{skin}_preview.1.jpg'
        source['observation']='local face-detail comparison; source names alone do not prove different scanned people'
        source['status']='existing local member, extracted for explicit face-detail revision'
        texture=out/(skin+'.png')
        with zipfile.ZipFile(source['archive']) as z:texture.write_bytes(z.read(source['member']))
        appearance=deepcopy(m['appearance']);appearance.update(body_texture=str(texture),character_id=cid)
        scalp(appearance,np.load(PILOT/'groom_reference/reference.npz'),out)
        m.update(character_id=cid,parent_asset_id=item['character_id'],parent_manifest=str(parent),version=version,
            texture_source=source,appearance=appearance,revision_reason='same-condition review: improve eyebrow/eyelid/nose/lip detail differences, not skin tone alone',
            geometry_reused_unchanged=True,status='body_frozen_revised_appearance_candidate')
        m['reference'].update(character_id=cid,appearance=appearance)
        for d in report['derivatives']:
            if d['character_id']==item['character_id']:
                d.update(character_id=cid,appearance=appearance)
                d['lineage']['appearance_source']=skin
        oldid=item['character_id'];item.update(character_id=cid,manifest=str(out/'manifest.json'));write_json(out/'manifest.json',m)
        old_comparison=json.loads(next((root/'uniform_inputs').glob(f'cycling_{short}_*/manifest.json')).read_text())
        comparison=deepcopy(m);comparison['reference']['clothing_geometry']=old_comparison['reference']['clothing_geometry']
        comparison['reference'].pop('groom_geometry',None)
        dest=uniform/cid;dest.mkdir();write_json(dest/'manifest.json',comparison)
        comparisons.append(dict(character_id=cid,manifest=str(dest/'manifest.json')))
        history.append(dict(identity_family_id=m['identity_family_id'],old_asset=oldid,new_asset=cid,
            old_status='needs_identity_detail_revision',new_status='review_pending',body_groom_clothing_changed=False,skin_source=source))
    write_json(uniform/'report.json',dict(characters=comparisons,derivatives=[],uniform_comparison=True,split='integration_only'))
    write_json(root/f'{a.revision}.json',dict(cases=history,original_assets_preserved=True,independent_identity_count_added=0))
    write_json(root/'report.json',report)

if __name__=='__main__':main()
