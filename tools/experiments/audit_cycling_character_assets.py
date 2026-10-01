#!/usr/bin/env python3
"""Inventory existing BEDLAM archives; copy only bundled review images/text.

No network, model evaluation, rendering, fitting, or asset extraction for use.
"""
import argparse
import collections
import csv
import io
import json
from pathlib import Path
import tarfile
import zipfile

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bedlam-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    with (repo / 'third_party/bedlam_render/config/gender.csv').open() as f:
        genders = {row['Name']: row['Gender'] for row in csv.DictReader(f)}
    selected = ['skin_f_asian_01_ALB', 'skin_f_african_01_ALB',
                'skin_m_white_01_ALB', 'skin_m_african_01_ALB']
    report = {'scope': 'local archive headers and existing previews only',
              'body_textures': {}, 'clothing_archives': []}
    archive = args.bedlam_root / 'body_textures/bedlam_body_textures_meshcapade.zip'
    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        (args.output / 'body_texture_members.txt').write_text(
            ''.join(f'{m.file_size}\t{m.filename}\n' for m in members))
        skins = [m for m in members if '/skin/skin_' in m.filename and m.filename.endswith('.png')]
        report['body_textures'] = dict(
            archive=str(archive), bytes=archive.stat().st_size,
            members=len(members), skin_counts=dict(collections.Counter(
                m.filename.split('/')[3] for m in skins)), selected=[])
        for texture in selected:
            member = next(m for m in skins if Path(m.filename).stem == texture)
            preview = next(m for m in members if Path(m.filename).name == f'{texture}_preview.1.jpg')
            preview_path = args.output / Path(preview.filename).name
            preview_path.write_bytes(z.read(preview))
            with z.open(member) as f, Image.open(f) as im:
                size, mode = list(im.size), im.mode
            report['body_textures']['selected'].append(dict(
                texture_id=texture, member=member.filename, bytes=member.file_size,
                size=size, mode=mode, preview_member=preview.filename,
                preview_copy=str(preview_path)))
        for name in ['readme.txt', 'Meshcapade_textures_CC-by-nc.md']:
            (args.output / name).write_bytes(z.read(f'bedlam_body_textures_meshcapade/{name}'))
    for kind in ['clothing_npz_6fps', 'clothing_textures']:
        archive = args.bedlam_root / 'clothing_data' / kind / f'{kind}.0.tar'
        with tarfile.open(archive, 'r:') as t:
            members = t.getmembers()
            subjects = sorted({m.name.split('/')[0] for m in members})
            (args.output / f'{kind}_members.txt').write_text(
                ''.join(f'{m.size}\t{m.name}\n' for m in members))
            item = dict(archive=str(archive), bytes=archive.stat().st_size,
                        members=len(members), file_extensions=dict(collections.Counter(
                            Path(m.name).suffix for m in members if m.isfile())),
                        subjects=[dict(id=s, gender=genders.get(s),
                                       source_person_group=s.split('_posed_')[0]) for s in subjects])
            if kind == 'clothing_npz_6fps':
                item['sample_schemas'] = []
                for subject in ['rp_aaron_posed_009', 'rp_alexandra_posed_002', 'rp_alison_posed_001']:
                    member = next(m for m in members if m.name.startswith(subject + '/') and m.name.endswith('.npz'))
                    with t.extractfile(member) as f, zipfile.ZipFile(io.BytesIO(f.read())) as npz:
                        item['sample_schemas'].append(dict(member=member.name, keys=npz.namelist()))
            report['clothing_archives'].append(item)
    (args.output / 'archive_inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'skin_counts': report['body_textures']['skin_counts'],
                      'clothing_subject_counts': [len(x['subjects']) for x in report['clothing_archives']]}, indent=2))


if __name__ == '__main__':
    main()
