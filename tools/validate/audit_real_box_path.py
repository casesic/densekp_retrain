"""Read a small saved dataset3 detector example; no detection, remapping or fitting."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--external',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    base=a.external/'data_process/tmp/hmr2_external_bbox_2214_densekp'
    args=json.loads((base/'debug/args.json').read_text())
    records=[];panels=[]
    for camera in ('cam2','cam4'):
        image_path=a.external/f'data_process/tmp/densekp_phaseA_erp_input/{camera}/02214.png'
        image=Image.open(image_path).convert('RGB');assert image.size==(5760,2880)
        overview=image.resize((1152,576));draw=ImageDraw.Draw(overview)
        for path in sorted((base/camera).glob('person_*.npz')):
            z=np.load(path);box=z['bbox_xyxy'].astype(float);center=(box[:2]+box[2:])/2;side=max(box[2:]-box[:2])
            draw.rectangle(tuple(box/5),outline='red',width=2);draw.text(tuple(box[:2]/5),path.stem,fill='white')
            crop=image.crop(tuple(np.r_[center-side/2,center+side/2])).resize((256,256))
            crop.save(out/f'{camera}_{path.stem}_square.png')
            crop.crop((32,0,224,256)).save(out/f'{camera}_{path.stem}_center.png')
            panel=Image.new('RGB',(256,280));panel.paste(crop,(0,24))
            ImageDraw.Draw(panel).text((3,4),f'{camera}/{path.stem} (qualitative)',fill='white');panels.append(panel)
            records.append(dict(camera=camera,source=str(path),image=str(image_path),bbox_xyxy=box.tolist(),
                crop_side=side,box_to_crop_expansion=1.,scope='saved detector output; no ground-truth accuracy or detector-error inference'))
        overview.save(out/f'{camera}_boxes.jpg')
    sheet=Image.new('RGB',(256*len(panels),280))
    for i,panel in enumerate(panels):sheet.paste(panel,(256*i,0))
    sheet.save(out/'real_inputs.jpg')
    report=dict(saved_run_args=args,boxes=records,
        verified_code=dict(detector='YOLOX-s HumanArt 640 end2end ONNX; rtmlib_yolox; saved threshold 0.25',
            preprocessing='BGR float32, resize min(640/h,640/w), bottom/right pad114; embedded NMS then class0/score threshold; divide ratio and clip to ROI',
            roi='Current component_bbox: strict two-camera overlap, periodic connected components, 160px dilation at business size; detector per cropped ROI',
            densekp='center=(xy1+xy2)/2; crop=max(width,height), no1.2; full256square normalized, model middle192; inverse affine then ROI offsets, x modulo width',
            outer_rotation='erp_rotate only after full-ERP restoration; new alias maps20260320; not20260311'),
        limitations=['One saved frame, five boxes, not a representative YOLO error distribution.',
            'No reliable full-body GT or temporal target association; no numerical detector error or DenseKP accuracy.',
            'Preview uses PIL square resampling for qualitative box context; synthetic acceptance uses actual Dataset affine.',
            'Legacy camerahmr densekp_npz boxes use portrait crop coordinates (e.g.343x612), not5760x2880 ERP; excluded from numeric comparisons.',
            'Current production also exposes TRT with exact ONNX NMS tail. Saved inspected run is ORT; do not infer all deployment runs share settings.'])
    (out/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
