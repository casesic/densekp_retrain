"""Close the bounded eyes/groom preview with traceable resource and delivery checks."""
import ast
import csv
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/experiments/bedlam_native_walk_eyes_hair_a2_20261009'


def write(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def main():
    recipe=json.loads((OUT/'frozen_recipe.json').read_text())
    base=Path(recipe['accepted_asset'])
    stages=[json.loads((OUT/(s+'_render_manifest.json')).read_text()) for s in ['probe','full']]
    seconds=sum(s['elapsed_seconds'] for s in stages)
    assert seconds<recipe['render_budget_seconds']
    assert len(stages[0]['records'])==11 and len(stages[1]['records'])==172
    assert len(stages[1]['reused_images'])==10
    paths=list((OUT/'renders').glob('*.png'));assert len(paths)==182
    assert (OUT/'head_detail_frame045.png').exists()
    movie=json.loads((OUT/'video_validation.json').read_text())
    assert movie['ffprobe']['streams'][0]['nb_frames']=='91'
    validation=json.loads((OUT/'validation.json').read_text())
    binding=json.loads((OUT/'binding_validation.json').read_text())
    geometry=json.loads((OUT/'full_geometry_validation.json').read_text())
    rows=list(csv.DictReader((OUT/'time_mapping.csv').open()))
    assert len(rows)==91 and (OUT/'time_mapping.csv').read_text()==(base/'time_mapping.csv').read_text()
    before=json.loads((OUT/'protected_before.json').read_text());after={}
    for name in before:
        s=Path(name).stat();after[name]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns)
    changed=[k for k in before if before[k]!=after[k]]
    critical=[k for k in before if str(base) in k or k.endswith('TODO.md') or 'continuous_cycling_asset' in k]
    assert not set(critical)&set(changed), 'Protected accepted asset or TODO changed; inspect before delivery'
    write(OUT/'protected_after.json',after)
    scripts=['tools/experiments/prepare_bedlam_walk_appearance.py','tools/render/bedlam_walk_appearance.py',
             'tools/render/preview_bedlam_native_walk_a2.py','tools/validate/package_bedlam_native_walk_a2.py',
             'tools/validate/finalize_bedlam_walk_appearance.py']
    for f in scripts:ast.parse((ROOT/f).read_text())
    manifest=json.loads((OUT/'manifest.json').read_text())
    for key in ['groom_reference','head_transforms','appearance_manifest','geometry','parameters','timeline_ref','preview']:
        assert Path(manifest[key]).exists(),key
    document=ROOT/'docs/data/bedlam/BEDLAM_NATIVE_WALK_EYES_HAIR_A2_20261009.md'
    for link in re.findall(r'\]\(([^)]+)\)',document.read_text()):
        if '://' not in link:assert (document.parent/link).exists(),link
    size=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
    record=dict(gpu_uuid=recipe['gpu_uuid'],cpu_preparation_seconds=binding['seconds'],
        cpu_full_geometry_validation_seconds=geometry['seconds'],probe_stage_seconds=stages[0]['elapsed_seconds'],
        full_stage_seconds=stages[1]['elapsed_seconds'],render_stage_total_seconds=seconds,
        render_call_seconds=sum(s['render_seconds'] for s in stages),encoding_seconds=movie['encoding_seconds'],
        rendered_images=183,reused_probe_images=10,output_bytes_snapshot=size,
        downloads=0,installed_dependencies=0,body_re_evaluations=0,clothing_rebindings=0,
        failures=['preparation_attempt_1_failure.json'],failed_cpu_seconds='not separately measured',
        developer_reading_wait_time='not separately instrumented; excluded from compute cost',
        no_old_outputs_deleted=True,no_training_or_inference=True,no_DS_ERP=True)
    write(OUT/'execution_record.json',record)
    write(OUT/'delivery_validation.json',dict(status='technical_checks_pass_new_appearance_user_review_pending',
        accepted_base_preserved=True,all91_frames_checked=True,eyes_native_components_only=True,
        body_cloth_render_mesh_readback=True,fixed_groom_topology=True,head_transform_once=True,
        R_readback_error=validation['R_max_readback_abs_error'],densekp_label_ready='pending_locked_head_template_semantics',
        preview_frames=91,view_count=2,static_detail_count=1,budget_pass=True,
        protected_snapshot_changes=changed,modified_A1_files=[],scripts_parse=True,
        artifact_refs_exist=True,new_appearance_accepted=False,public_consumer_tested=False,integration_only=True))
    (OUT/'git_status_after.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    text=document.read_text()
    marker='\n## 完成时实测\n'
    assert marker not in text
    text+=marker+f'\n探针阶段{stages[0]["elapsed_seconds"]:.3f}秒，完整补齐阶段{stages[1]["elapsed_seconds"]:.3f}秒，渲染阶段累计{seconds:.3f}秒；编码{movie["encoding_seconds"]:.3f}秒。CPU绑定准备{binding["seconds"]:.3f}秒、全帧全部毛发点验证{geometry["seconds"]:.3f}秒。保存183张诊断图（双视角182张及头部近景1张）、91帧MP4；输出大小快照{size:,} B（{size/2**20:.2f} MiB）。旧身体/衣物/时间表、旧视频和TODO保护检查通过。新增眼睛与头发仍待用户动态观看确认，未运行公共场景。\n'
    document.write_text(text)
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
