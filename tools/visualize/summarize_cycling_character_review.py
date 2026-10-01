#!/usr/bin/env python3
"""Arrange actual review renders, without changing their content or color."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def sheet(rows, output, tile=400):
    image=Image.new('RGB',(len(rows[0])*tile,len(rows)*(tile+32)),(35,35,35))
    draw=ImageDraw.Draw(image)
    for y,row in enumerate(rows):
        for x,(path,label) in enumerate(row):
            panel=Image.open(path).convert('RGB');panel.thumbnail((tile,tile))
            left=x*tile+(tile-panel.width)//2;top=y*(tile+32)
            image.paste(panel,(left,top));draw.text((x*tile+5,top+tile+5),label,fill='white')
    image.save(output,quality=95)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('assets',type=Path);p.add_argument('review',type=Path)
    a=p.parse_args();report=json.loads((a.assets/'report.json').read_text())
    chars=[x['character_id'] for x in report['characters']]
    for name,views in [('identity_heads',['head_front','head_threequarter']),
                       ('shape_comparison',['front','head_clay'])]:
        rows=[[(a.review/c/'reference'/f'{v}.png',f'{c} / {v}') for c in chars] for v in views]
        sheet(rows,a.review/f'{name}.jpg',480)
    poses=sorted({d['pose_source_id'] for d in report['derivatives']})
    for i,pose in enumerate(poses,1):
        rows=[[(a.review/c/pose/f'{v}.png',f'{c} / P{i} / {v}') for v in ['front','side','back']] for c in chars]
        sheet(rows,a.review/f'pose_P{i}_multiview.jpg')


if __name__=='__main__':main()
