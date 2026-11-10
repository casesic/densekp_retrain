"""Application-side ROI raster metadata and native box support validation."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from densekp.core.datasets.image_layout import LAYOUT_FIELDS,parse_image_layout
from .boxes import validate_box_context,crop_from_box
from densekp_retrain.imaging.regions import require_crop_in_region


def roi_metadata(roi,full_size,physical_mask_ref):
    x0,y0,x1,y1=roi
    row=dict(image_layout='roi_image',image_layout_version=1,full_image_size_wh=list(full_size),roi_xyxy=list(roi),
        offset_xy=[x0,y0],stored_size_wh=[x1-x0,y1-y0],physical_mask_ref=str(physical_mask_ref),
        render_coverage_xyxy=list(roi),outside_stored_region='unknown',image_filename='roi.png')
    parse_image_layout(row)
    return row


def validate_roi_box(box,entry,person_id,annotation_person_id,meta,valid,person_mask):
    layout=parse_image_layout(meta)
    if layout is None or layout.kind!='roi_image':raise ValueError('required_roi_image_layout')
    local=np.asarray(box)-np.tile(layout.offset_xy,2)
    validate_box_context(local,bbox_kind='visible_instance_tight',bbox_source=entry['source'],person_id=person_id,
        mask_person_id=entry['person_id'],annotation_person_id=annotation_person_id,
        image_size_wh=layout.stored_size_wh,manifest_size_wh=layout.stored_size_wh,valid_mask=valid,person_mask=person_mask,
        boundary_policy='raster_clip',min_side_px=2)
    center,scale=crop_from_box(box,image_size_wh=layout.full_size_wh,crop_policy='deployment',boundary_policy='erp_reject_seam',min_side_px=2)
    require_crop_in_region(center,scale,layout.roi_xyxy,layout.full_size_wh)
    return center,scale


def save_roi_observation(dest,rgb,instance,data,valid,meta):
    """Store only local masks; annotation XY and native boxes remain full ERP."""
    dest=Path(dest);layout=parse_image_layout(meta)
    if tuple(rgb.shape[:2][::-1])!=layout.stored_size_wh or instance.shape!=valid.shape or valid.shape!=rgb.shape[:2]:
        raise ValueError('ROI_raster_size_conflict')
    Image.fromarray(rgb).save(dest/'roi.png');Image.fromarray(valid.astype(np.uint8)*255).save(dest/'valid_mask.png')
    np.savez(dest/'annotations.npz',**data)
    boxes=[]
    for pid in data['person_ids']:
        pm=(np.rint(instance)==pid)&valid
        Image.fromarray(pm.astype(np.uint8)*255).save(dest/f'person_{pid}_mask.png')
        y,x=np.where(pm)
        local=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1] if len(x) else None
        box=(np.array(local)+np.tile(layout.offset_xy,2)).tolist() if local else None
        boxes.append(dict(person_id=int(pid),bbox_xyxy_exclusive=box,source='oracle rendered ROI instance mask + full ERP integer offset; not detector output'))
    for name,value in [('boxes',boxes),('raster_storage',meta),('render_coverage',dict(full_image_size_wh=list(layout.full_size_wh),
            high_quality_region_xyxy=list(layout.roi_xyxy),label_coverage_region_xyxy=list(layout.roi_xyxy),
            outside_region_depth_and_ID='unknown',person_mask_zero_outside_region='unknown, not certified background',physical_valid_mask_ref=meta['physical_mask_ref']))]:
        (dest/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
