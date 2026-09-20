#!/usr/bin/env python3
"""生成 P0 裸模调试基线与最小外观基线的并排对照图。"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

def main():
    p=argparse.ArgumentParser(); p.add_argument('--before',type=Path,required=True); p.add_argument('--after',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--left-label',default='Before: naked/debug baseline'); p.add_argument('--right-label',default='After: material + ground + world + 3-point light'); a=p.parse_args()
    left=Image.open(a.before).convert('RGB'); right=Image.open(a.after).convert('RGB')
    if right.size!=left.size: right=right.resize(left.size)
    bar=44; canvas=Image.new('RGB',(left.width*2,left.height+bar),(18,18,18)); canvas.paste(left,(0,bar)); canvas.paste(right,(left.width,bar)); d=ImageDraw.Draw(canvas)
    d.text((12,14),a.left_label,fill=(255,255,255)); d.text((left.width+12,14),a.right_label,fill=(255,255,255))
    a.output.parent.mkdir(parents=True,exist_ok=True); canvas.save(a.output)

if __name__=='__main__': main()
