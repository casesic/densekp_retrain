#!/usr/bin/env python3
"""Select explicit V-COCO ride+bicycle pairs and download a small COCO image pilot."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vcoco', type=Path, required=True)
    parser.add_argument('--coco-annotations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')

    actions = json.loads(args.vcoco.read_text())
    ride = next(row for row in actions if row['action_name'] == 'ride')
    # Official V-COCO stores one flattened column per role.
    roles = np.array(ride['role_object_id']).reshape(len(ride['role_name']), -1).T
    np.testing.assert_array_equal(roles[:, 0], ride['ann_id'])
    instrument = ride['role_name'].index('instr')
    wanted_images = {image for image, label in zip(ride['image_id'], ride['label']) if label == 1}
    annotations, images, licenses, categories = {}, {}, {}, {}
    with zipfile.ZipFile(args.coco_annotations) as archive:
        for split in ('train', 'val'):
            data = json.loads(archive.read(f'annotations/instances_{split}2014.json'))
            annotations.update({a['id']: a for a in data['annotations'] if a['image_id'] in wanted_images})
            images.update({i['id']: i for i in data['images'] if i['id'] in wanted_images})
            licenses.update({i['id']: i for i in data['licenses']})
            categories.update({i['id']: i['name'] for i in data['categories']})
            keypoints = json.loads(archive.read(f'annotations/person_keypoints_{split}2014.json'))
            for person in keypoints['annotations']:
                if person['id'] in annotations:
                    annotations[person['id']]['keypoints'] = person['keypoints']

    candidates = []
    for index, label in enumerate(ride['label']):
        if label != 1 or roles[index, instrument] == 0:
            continue
        person = annotations[int(roles[index, 0])]
        bike = annotations[int(roles[index, instrument])]
        if categories[bike['category_id']] != 'bicycle':
            continue
        image = images[person['image_id']]
        if categories[person['category_id']] != 'person' or bike['image_id'] != image['id']:
            raise ValueError('V-COCO person/instrument identity mismatch')
        x, y, w, h = person['bbox']
        if person['iscrowd'] or min(w, h) < 100 or h < 180:
            continue
        if x < 2 or y < 2 or x+w > image['width']-2 or y+h > image['height']-2:
            continue
        candidates.append(dict(
            sample_id=f"coco_{image['id']}_person_{person['id']}",
            image_id=image['id'], person_annotation_id=person['id'], bicycle_annotation_id=bike['id'],
            image_file=f"images/{image['file_name']}", width=image['width'], height=image['height'],
            bbox_xyxy=[x, y, x+w, y+h], bbox_xywh=person['bbox'], bicycle_bbox_xywh=bike['bbox'],
            coco_keypoints17=np.array(person['keypoints']).reshape(17, 3).tolist(),
            source_coco_url=image['coco_url'], flickr_url=image.get('flickr_url'),
            image_license=licenses[image['license']],
            vcoco_split='train', coco_split=image['file_name'].split('_')[1],
            evidence='V-COCO ride=1; instrument COCO category=bicycle; person bbox from COCO',
            review_status='pending visual review',
        ))
    candidates.sort(key=lambda row: (-row['bbox_xywh'][2]*row['bbox_xywh'][3], row['sample_id']))
    chosen, seen = [], set()
    for row in candidates:
        if row['image_id'] not in seen:
            chosen.append(row)
            seen.add(row['image_id'])
        if len(chosen) == args.limit:
            break
    if not chosen:
        raise ValueError('No eligible cycling images')

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'images').mkdir()
    shutil.copyfile(args.vcoco, args.output / 'vcoco_train.json')
    (args.output / 'candidates.json').write_text(json.dumps(candidates, indent=2) + '\n')
    sheet = Image.new('RGB', (4*320, ((len(chosen)+3)//4)*280), 'white')
    for index, row in enumerate(chosen):
        # Same official COCO S3 bucket, with a hostname covered by its TLS certificate.
        suffix = row['source_coco_url'].split('images.cocodataset.org/', 1)[1]
        url = 'https://s3.amazonaws.com/images.cocodataset.org/' + suffix
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        path = args.output / row['image_file']
        path.write_bytes(response.content)
        with Image.open(path) as source:
            if source.size != (row['width'], row['height']):
                raise ValueError(f'Unexpected raster size: {path}')
            canvas = source.convert('RGB')
        ImageDraw.Draw(canvas).rectangle(row['bbox_xyxy'], outline='red', width=3)
        panel = ImageOps.pad(canvas, (320, 250), color='white')
        sheet.paste(panel, ((index % 4)*320, (index//4)*280+25))
        ImageDraw.Draw(sheet).text(((index % 4)*320+5, (index//4)*280+5), row['sample_id'], fill='black')
        row.update(download_url=url, image_bytes=path.stat().st_size)
        print(row['sample_id'], flush=True)
    sheet.save(args.output / 'image_review.jpg')
    manifest = dict(
        dataset='V-COCO train with COCO 2014 images', samples=chosen,
        eligible_person_bicycle_pairs=len(candidates), downloaded_images=len(chosen),
        selection='ride+bicycle; noncrowd, bbox min side 100px, height 180px, 2px image margin; largest area; one target/image',
        sequence=False, purpose='independent pose acquisition pilot, not a held-out HMR benchmark',
        sources=['https://github.com/s-gupta/v-coco', 'https://cocodataset.org/'],
    )
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
