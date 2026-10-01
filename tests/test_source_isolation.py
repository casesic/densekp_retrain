import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools.experiments.audit_identity_texture_sources import components
from tools.validate.validate_source_isolated_split import validate_split


class SourceIsolationTests(unittest.TestCase):
    def test_excluding_bridge_does_not_discard_unrelated_families(self):
        edges=[('F01','F04'),('F02','F04'),('F01','F05')]
        self.assertEqual(components(['F01','F02','F04','F05'],edges),[['F01','F02','F04','F05']])
        self.assertEqual(components(['F01','F02','F05'],edges),[['F01','F05'],['F02']])

    def test_unreviewed_old_split_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'facial-source audit'):
            validate_split(dict(status='frozen_before_new_combinations'))

    def test_possible_source_and_occluder_leakage_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audit=root/'audit.json'
            audit.write_text(json.dumps(dict(conservative_possible_source_groups={'x':['F01'],'y':['F02']})))
            chars=[dict(character_id=n,identity_family_id='cycling_'+n,partition=s,gender=g,
                face_source_group=n,shape_source_group=n) for n,s,g in
                [('F01','train','female'),('M01','train','male'),('F02','val','female'),('M02','val','male')]]
            split=dict(status='frozen_before_new_combinations',source_audit=str(audit),characters=chars,
                excluded_families=[],source_groups={s:dict(cycling=s+'photo',pedestrian=s+'clip',pedestrian_person=s+'actor') for s in ['train','val']})
            self.assertEqual(validate_split(split)['status'],'passed')
            bad=copy.deepcopy(split);bad['characters'][2]['shape_source_group']='F01'
            with self.assertRaisesRegex(ValueError,'shape_source_group'):validate_split(bad)
            scene=root/'scene';scene.mkdir();(scene/'people.json').write_text(json.dumps(dict(people=[
                dict(character_id='F01',action='cycling',source_group='trainphoto'),
                dict(character_id='M02',action='walking',source_group='valclip')])))
            row=dict(quality_state='candidate_accepted',split='train',scene=str(scene),image=str(scene/'cam2.png'))
            with self.assertRaisesRegex(ValueError,'occluder identity'):
                validate_split(split,{'train':dict(split='train',frozen_utc='fixed',rows=[row])})

if __name__=='__main__':unittest.main()
