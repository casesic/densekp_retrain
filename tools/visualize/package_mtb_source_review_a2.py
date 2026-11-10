"""Package the bounded source audit. No invented media time and no inference."""
from pathlib import Path
import html
import json
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/audits/mtb_cycling_pose_dataset_a2_20261009'


def dump(name, obj):
    p = OUT / name
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    start=time.perf_counter(); cpu=time.process_time()
    tasks=json.loads((OUT/'annotation_mapping.json').read_text())
    stats=json.loads((OUT/'annotation_statistics.json').read_text())
    observations=json.loads((OUT/'review_findings.json').read_text())
    candidates=[]
    for seq in [4,12]:
        task=tasks[seq-1]; sid=task['sequence_id']; note=observations[sid]
        execution=json.loads((OUT/f'sequence_{sid}_execution.json').read_text())
        assert execution['successful_images']==24
        assert not any('error' in r for r in execution['results'])
        frames=task['frames']
        for f in frames:
            assert (OUT/f['image']).is_file()
        mapping=[{k:v for k,v in f.items() if k not in ['points','head_box']} for f in frames]
        dump(f'sequence_{sid}_time_mapping.json',{
            'classification':'ordered_sparse_extracted_images_unknown_time',
            'source_fps':None,'source_time_base':None,'source_interval':None,
            'display_interval_ms':500,'display_interval_is_source_time':False,
            'basis':'XML global frame numbers match published image names; not original-video frame IDs',
            'rows':mapping})
        candidate={
            'candidate_id':f'CyclingPose20800829_v1_seq{sid}',
            'scope':'integration_only','asset_ready':False,'continuous_fit_ready':False,
            'dataset_url':'https://zenodo.org/records/20800829','doi':'10.5281/zenodo.20800829',
            'version':'v1','sequence_id':sid,'task_id':task['task_id'],
            'dataset_internal_source_name':task['source_name'],'dataset_person_id':None,
            'source_post_url':None,'original_video_url':None,'source_author_identifier':None,
            'local_original_video':None,'original_video_available':'unknown_not_identified',
            'dataset_license':'CC-BY-4.0','license_evidence':'license_evidence.json',
            'original_video_download_reconstruction_training_permission':'unknown',
            'dataset_derivative_training_basis':'published CC BY4.0 with attribution/change notice; upstream rights chain not independently supplied',
            'type':'mountain_bike_visually_supported','visual_evidence':note,
            'image_size':task['size'],'annotation_statistics':stats[seq-1],
            'available_sample_count':24,'time_mapping':f'sequence_{sid}_time_mapping.json',
            'source_fps':None,'source_time_base':None,'source_start_seconds':None,'source_end_seconds':None,
            'source_start_frame':None,'source_end_frame':None,
            'temporal_class':'sparse_ordered_images_unknown_original_timing',
            'publisher_cycle_claim':2,'verified_original_continuous_cycles':None,
            'image_processing':{'face_anonymization':'publisher retained','crop_resize_pipeline':'unknown',
                                'image_header_equals_task_size':True,'PNG_time_metadata':None},
            'source_group':'CyclingPose_Zenodo20800829_reddit_identity_unknown',
            'subgroup':f'internal_video_{sid}_person_unknown','official_split':'default_no_train_val_evidence',
            'independent_identity_verified':False,
            'recommendation':'first_choice_for_provenance_followup_not_fitting' if seq==4 else 'backup_sparse_pose_reference',
            'missing':['trusted original-post/video mapping','lawfully available original clip',
                       'original PTS/fps and frame/cycle mapping','upstream video reuse authority'],
            'next_clip':{'target_seconds':[3,5],'actual_start_seconds':None,'actual_end_seconds':None,
                         'condition':'only after original timing and continuous clip verified; do not loop/interpolate these images'},
            'review_status':'assistant inspected source images/contact sheets; no real-time original-video viewing; user posture review pending',
            'preview_contact':f'sequence_{sid}_contact.jpg','annotation_contact':f'sequence_{sid}_annotations.jpg',
            'frames':mapping}
        candidates.append(candidate)
    dump('candidates.json',{'schema':'mtb_source_candidates_a2_v1','scope':'integration_only',
         'recommended_id':candidates[0]['candidate_id'],'qualified_continuous_video_count':0,
         'confirmed_visual_mtb_candidates':2,'candidates':candidates})
    cards=[]
    for c in candidates:
        sid=c['sequence_id']; n=c['visual_evidence']
        cards.append(f'''<section id="seq{sid}"><h2>序列 {sid} · {'首选补证' if sid=='04' else '备选'}</h2>
<p>{html.escape(n['summary'])}</p><p><b>原视频：未定位。原速：未知。不能开始连续拟合。</b></p>
<dl><dt>车型证据</dt><dd>{html.escape(n['bike_evidence'])}</dd>
<dt>来源与时间</dt><dd>Zenodo v1，task {c['task_id']}，{html.escape(c['dataset_internal_source_name'])}；24张抽帧。原帖子/作者账号/fps/PTS均未提供，不能猜配。</dd>
<dt>拍摄与问题</dt><dd>{html.escape(n['quality'])}</dd>
<dt>与 APD6526 的差异</dt><dd>{html.escape(n['apd_difference'])}</dd>
<dt>许可</dt><dd>发布图像/标注 CC BY4.0；未发布原视频的许可 unknown。<a href="license_evidence.json">证据</a></dd></dl>
<div class="warning">抽帧顺序预览，非原始播放速度。每张停留500ms仅为方便浏览，不代表源fps；不插值、不补帧、不循环。</div>
<div class="viewer" data-seq="{sid}"><canvas aria-label="匿名化原图与可选标注"></canvas>
<p class="frameinfo"></p><div class="controls"><button class="prev">上一张</button><button class="play">顺序浏览一次</button><button class="next">下一张</button>
<input class="slider" aria-label="抽帧序号" type="range" min="0" max="23" value="0"><label><input type="checkbox" class="overlay">显示2D标注</label></div>
<p>绿色：可见标注；橙色：遮挡标注。圆点：keyframe=1；方框：插值点。标注不是DenseKP138或3D真值。<a class="original" href="{c['frames'][0]['image']}">打开当前原图</a></p></div>
<details><summary>24张联系表与逐图时间空缺</summary><a href="{c['preview_contact']}"><img loading="lazy" src="{c['preview_contact']}" alt="全部24张抽帧联系表"></a>
<p><a href="{c['annotation_contact']}">标注联系表</a> · <a href="{c['time_mapping']}">逐图映射（原时间null）</a></p></details></section>''')
    payload=json.dumps({t['sequence_id']:t for t in tasks if t['sequence_id'] in ['04','12']},ensure_ascii=False)
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>山地车动作来源评审 · A2</title><style>
body{max-width:1120px;margin:32px auto;padding:0 20px;background:#10151d;color:#e6edf6;font:17px/1.65 system-ui,sans-serif}a{color:#8ecaff}h1,h2{line-height:1.3}section{margin:32px 0;padding:24px;border:1px solid #3d4a5c;border-radius:14px;background:#17202b}.warning{background:#503a1f;padding:12px;border-radius:8px}dt{font-weight:bold;color:#adc5df}dd{margin:0 0 12px}canvas{display:block;max-width:100%;max-height:620px;width:auto;height:auto;margin:12px auto;background:#080b10}img{max-width:100%}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}button{padding:9px;background:#d8eaff;color:#17202b;border:0;border-radius:5px;cursor:pointer}.slider{flex:1;min-width:150px}.muted{color:#b2becb}table{border-collapse:collapse;width:100%}td,th{text-align:left;border-bottom:1px solid #44505f;padding:8px}
</style><h1>山地车连续动作：先评审来源</h1>
<p>2026-10-09 · integration_only · <b>2个山地车图像候选，0个已取得且时间可靠的原视频。</b></p>
<p>建议先看<a href="#seq04">序列04</a>；<a href="#seq12">序列12</a>为备选。可判断姿态是否符合需求，但本页不能证明原始速度或连续重建已就绪。两段均为骑行台来源，未来道路行进由场景系统负责。</p>
<p><a href="https://zenodo.org/records/20800829">官方来源</a> · <a href="candidates.json">候选清单</a> · <a href="data_definition.json">标注定义</a> · <a href="execution_record.json">下载/CPU记录</a> · <a href="../../../docs/motion/cycling/MTB_CYCLING_POSE_DATASET_A2_20261009.md">完整报告</a></p>
''' + ''.join(cards) + '''<section><h2>20组首图筛选留痕</h2><p>每组只用首图辨认车型；仅04/12展开24张。10为平把硬叉/窄胎，山地车型证据不足；11原图可见弯把。没有把未选序列的动态质量判为失败。</p><a href="screen_contact.jpg"><img loading="lazy" src="screen_contact.jpg" alt="20组首图筛选"></a><a href="screening_findings.json">逐组未选原因</a></section>
<p class="muted">署名：Matthew Vella、Kristian Guillaumier，University of Malta，Cycling Pose Estimation Dataset，v1，2026，DOI10.5281/zenodo.20800829。发布素材<a href="https://creativecommons.org/licenses/by/4.0/">CC BY4.0</a>。原匿名化保留；联系表缩放/加标签、网页可选标注为本轮修改。<a href="ATTRIBUTION.md">完整说明</a>。</p>
<script id="source-data" type="application/json">''' + payload + '''</script><script>
const data=JSON.parse(document.getElementById('source-data').textContent);
document.querySelectorAll('.viewer').forEach(v=>{
 const task=data[v.dataset.seq],canvas=v.querySelector('canvas'),ctx=canvas.getContext('2d');
 const slider=v.querySelector('.slider'),play=v.querySelector('.play'),overlay=v.querySelector('.overlay');
 let i=0,running=false,timer=null,token=0;
 function stop(){running=false;clearTimeout(timer);play.textContent='顺序浏览一次';}
 function draw(){const current=++token,row=task.frames[i],img=new Image();slider.value=i;
  v.querySelector('.frameinfo').textContent=`抽帧 ${i+1}/24 · XML/图像编号 ${row.global_image_index} · 源帧、源时间：未知`;
  v.querySelector('.original').href=row.image;
  img.onload=()=>{if(current!==token)return;canvas.width=img.width;canvas.height=img.height;ctx.drawImage(img,0,0);
   if(overlay.checked) row.points.forEach(p=>{const[x,y]=p.xy;ctx.strokeStyle=p.occluded==='1'?'#ffab40':'#20ff70';ctx.lineWidth=3;ctx.beginPath();if(p.keyframe==='1')ctx.arc(x,y,6,0,Math.PI*2);else ctx.rect(x-6,y-6,12,12);ctx.stroke();});
   if(running)timer=setTimeout(()=>{if(i===23){stop();return;}i++;draw();},500);
  };img.onerror=()=>{stop();v.querySelector('.frameinfo').textContent+=' · 图像读取失败，不跳过';};img.src=row.image;
 }
 slider.oninput=()=>{stop();i=Number(slider.value);draw();};
 v.querySelector('.prev').onclick=()=>{stop();i=Math.max(0,i-1);draw();};
 v.querySelector('.next').onclick=()=>{stop();i=Math.min(23,i+1);draw();};
 play.onclick=()=>{if(running){stop();return;}if(i===23)i=0;running=true;play.textContent='停止浏览';draw();};
 overlay.onchange=()=>{stop();draw();};draw();
});
</script></html>'''
    with (OUT/'index.html').open('x') as f:f.write(document)
    dump('package_execution.json',{'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
        'original_video_count':0,'generated_video_count':0,'ordered_image_viewers':2,
        'display_ms':500,'display_is_original_timing':False,'interpolation_or_frame_duplication':False,
        'continuous_human_review':'not performed; no original video acquired'})


if __name__=='__main__':main()
